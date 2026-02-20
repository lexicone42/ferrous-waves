use crate::utils::error::Result;

/// Perceptual audio metrics based on human hearing characteristics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerceptualMetrics {
    /// Integrated loudness in LUFS (Loudness Units Full Scale)
    pub loudness_lufs: f32,

    /// Loudness range (LRA) in LU
    pub loudness_range: f32,

    /// True peak in dBFS
    pub true_peak_dbfs: f32,

    /// Dynamic range in dB
    pub dynamic_range: f32,

    /// Peak to average ratio (crest factor)
    pub crest_factor: f32,

    /// Perceived energy level (0.0 to 1.0)
    pub energy_level: f32,

    /// Short-term loudness values (3-second windows)
    pub short_term_loudness: Vec<f32>,

    /// Momentary loudness values (400ms windows)
    pub momentary_loudness: Vec<f32>,

    /// Shannon entropy of LUFS histogram (0 = constant loudness, 1 = maximally varied)
    pub dynamics_entropy: f32,

    /// Linear regression slope of LUFS over time (LUFS/minute; positive = crescendo)
    pub dynamics_slope: f32,

    /// Number of local loudness maxima with ≥3 LU prominence
    pub dynamics_peak_count: u32,
}

/// ITU-R BS.1770-4 K-weighting filter for LUFS measurement
struct KWeightingFilter {
    // Pre-filter (high-pass) coefficients
    pre_b: [f64; 3],
    pre_a: [f64; 3],
    // RLB filter (high-frequency shelf) coefficients
    rlb_b: [f64; 3],
    rlb_a: [f64; 3],
}

impl KWeightingFilter {
    fn new(sample_rate: f32) -> Self {
        // Pre-filter: 2nd order high-pass filter
        // Approximates human hearing's reduced sensitivity to low frequencies
        let f0 = 1681.974450955533;
        let q = 0.7071752369554196;
        let k = (std::f64::consts::PI * f0 / sample_rate as f64).tan();
        let k2 = k * k;
        let vq = k / q;
        let norm = 1.0 / (1.0 + vq + k2);

        let pre_b = [norm, -2.0 * norm, norm];
        let pre_a = [1.0, 2.0 * (k2 - 1.0) * norm, (1.0 - vq + k2) * norm];

        // RLB filter: High-frequency shelf filter
        let fh = 38.13547087602444;
        let q_h = 0.5003270373238773;
        let kh = (std::f64::consts::PI * fh / sample_rate as f64).tan();
        let kh2 = kh * kh;
        let vqh = kh / q_h;
        let normh = 1.0 / (1.0 + vqh + kh2);

        let rlb_b = [(vqh + kh2) * normh, 2.0 * kh2 * normh, kh2 * normh];
        let rlb_a = [1.0, 2.0 * (kh2 - 1.0) * normh, (1.0 - vqh + kh2) * normh];

        Self {
            pre_b,
            pre_a,
            rlb_b,
            rlb_a,
        }
    }

    /// Apply K-weighting filter to audio samples
    fn apply(&self, samples: &[f32]) -> Vec<f32> {
        let mut filtered = vec![0.0f32; samples.len()];

        // State for pre-filter
        let mut pre_x1 = 0.0;
        let mut pre_x2 = 0.0;
        let mut pre_y1 = 0.0;
        let mut pre_y2 = 0.0;

        // State for RLB filter
        let mut rlb_x1 = 0.0;
        let mut rlb_x2 = 0.0;
        let mut rlb_y1 = 0.0;
        let mut rlb_y2 = 0.0;

        for (i, &sample) in samples.iter().enumerate() {
            let x = sample as f64;

            // Apply pre-filter
            let pre_y = self.pre_b[0] * x + self.pre_b[1] * pre_x1 + self.pre_b[2] * pre_x2
                - self.pre_a[1] * pre_y1
                - self.pre_a[2] * pre_y2;

            pre_x2 = pre_x1;
            pre_x1 = x;
            pre_y2 = pre_y1;
            pre_y1 = pre_y;

            // Apply RLB filter
            let rlb_y = self.rlb_b[0] * pre_y + self.rlb_b[1] * rlb_x1 + self.rlb_b[2] * rlb_x2
                - self.rlb_a[1] * rlb_y1
                - self.rlb_a[2] * rlb_y2;

            rlb_x2 = rlb_x1;
            rlb_x1 = pre_y;
            rlb_y2 = rlb_y1;
            rlb_y1 = rlb_y;

            filtered[i] = rlb_y as f32;
        }

        filtered
    }
}

/// Calculate perceptual metrics from audio samples
pub fn calculate_perceptual_metrics(
    samples: &[f32],
    channels: usize,
    sample_rate: f32,
) -> Result<PerceptualMetrics> {
    // Ensure we have audio data
    if samples.is_empty() {
        return Ok(PerceptualMetrics {
            loudness_lufs: -70.0,
            loudness_range: 0.0,
            true_peak_dbfs: -70.0,
            dynamic_range: 0.0,
            crest_factor: 0.0,
            energy_level: 0.0,
            short_term_loudness: vec![],
            momentary_loudness: vec![],
            dynamics_entropy: 0.0,
            dynamics_slope: 0.0,
            dynamics_peak_count: 0,
        });
    }

    // Convert interleaved samples to mono for LUFS calculation
    let mono_samples = if channels > 1 {
        let frame_count = samples.len() / channels;
        let mut mono = vec![0.0f32; frame_count];
        for (i, frame) in mono.iter_mut().enumerate().take(frame_count) {
            let frame_start = i * channels;
            let frame_sum: f32 = samples[frame_start..frame_start + channels].iter().sum();
            *frame = frame_sum / channels as f32;
        }
        mono
    } else {
        samples.to_vec()
    };

    // Apply K-weighting filter
    let k_filter = KWeightingFilter::new(sample_rate);
    let weighted = k_filter.apply(&mono_samples);

    // Calculate integrated loudness (LUFS) with EBU R 128 gating
    let integrated_lufs = calculate_integrated_loudness(&weighted, sample_rate);

    // Calculate true peak
    let true_peak = mono_samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

    let true_peak_dbfs = if true_peak > 0.0 {
        20.0 * true_peak.log10()
    } else {
        -70.0
    };

    // Calculate RMS for dynamic range
    let rms = (mono_samples.iter().map(|&s| s * s).sum::<f32>() / mono_samples.len() as f32).sqrt();

    let rms_db = if rms > 0.0 { 20.0 * rms.log10() } else { -70.0 };

    // Dynamic range (peak to RMS difference)
    let dynamic_range = true_peak_dbfs - rms_db;

    // Crest factor (peak to average ratio)
    let crest_factor = if rms > 0.0 { true_peak / rms } else { 0.0 };

    // Calculate loudness range (LRA)
    let (loudness_range, short_term_loudness) = calculate_loudness_range(&weighted, sample_rate);

    // Calculate momentary loudness (400ms windows)
    let momentary_loudness = calculate_momentary_loudness(&weighted, sample_rate);

    // Calculate perceived energy level (0.0 to 1.0)
    // Map LUFS to energy: -60 LUFS = 0.0, -5 LUFS = 1.0
    let energy_level = ((integrated_lufs + 60.0) / 55.0).clamp(0.0, 1.0);

    // Dynamics trajectory features (computed from short_term_loudness)
    let (dynamics_entropy, dynamics_slope, dynamics_peak_count) =
        compute_dynamics_trajectory(&short_term_loudness);

    Ok(PerceptualMetrics {
        loudness_lufs: integrated_lufs,
        loudness_range,
        true_peak_dbfs,
        dynamic_range,
        crest_factor,
        energy_level,
        short_term_loudness,
        momentary_loudness,
        dynamics_entropy,
        dynamics_slope,
        dynamics_peak_count,
    })
}

/// Calculate integrated loudness according to EBU R 128
fn calculate_integrated_loudness(weighted: &[f32], _sample_rate: f32) -> f32 {
    // First pass: absolute gating at -70 LUFS
    let absolute_gate_linear = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
    let mut sum_squares = 0.0f64;
    let mut gated_samples = 0;

    for &sample in weighted.iter() {
        let square = (sample as f64) * (sample as f64);
        if square >= absolute_gate_linear {
            sum_squares += square;
            gated_samples += 1;
        }
    }

    let mean_square_gated = if gated_samples > 0 {
        sum_squares / gated_samples as f64
    } else {
        1e-10 // Minimum value to avoid log(0)
    };

    // Second pass: relative gating at -10 LU below ungated loudness
    let ungated_loudness = -0.691 + 10.0 * mean_square_gated.log10();
    let relative_gate_linear = 10.0_f64.powf((ungated_loudness - 10.0 + 0.691) / 10.0);

    let mut relative_sum_squares = 0.0f64;
    let mut relative_gated_samples = 0;

    for &sample in weighted.iter() {
        let square = (sample as f64) * (sample as f64);
        if square >= relative_gate_linear.max(absolute_gate_linear) {
            relative_sum_squares += square;
            relative_gated_samples += 1;
        }
    }

    let final_mean_square = if relative_gated_samples > 0 {
        relative_sum_squares / relative_gated_samples as f64
    } else {
        mean_square_gated
    };

    // Convert to LUFS using the EBU R 128 formula
    (-0.691 + 10.0 * final_mean_square.log10()) as f32
}

/// Calculate loudness range (LRA) and short-term loudness values
fn calculate_loudness_range(weighted: &[f32], sample_rate: f32) -> (f32, Vec<f32>) {
    let window_size = (sample_rate * 3.0) as usize; // 3-second windows
    let hop_size = (sample_rate * 0.1) as usize; // 100ms hop

    if weighted.len() < window_size {
        return (0.0, vec![]);
    }

    let mut window_loudnesses = Vec::new();
    let mut window_start = 0;

    while window_start + window_size <= weighted.len() {
        let window_end = window_start + window_size;
        let window_samples = &weighted[window_start..window_end];

        // Calculate mean square for this window
        let window_mean_square = window_samples
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / window_samples.len() as f64;

        // Apply absolute gating
        let absolute_gate_linear = 10.0_f64.powf((-70.0 + 0.691) / 10.0);
        if window_mean_square >= absolute_gate_linear {
            let window_loudness = -0.691 + 10.0 * window_mean_square.log10();
            window_loudnesses.push(window_loudness as f32);
        }

        window_start += hop_size;
    }

    if window_loudnesses.len() < 2 {
        return (0.0, window_loudnesses);
    }

    // Calculate LRA as difference between 95th and 10th percentiles
    let mut sorted_loudnesses = window_loudnesses.clone();
    sorted_loudnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let low_idx = (sorted_loudnesses.len() as f32 * 0.10) as usize;
    let high_idx = (sorted_loudnesses.len() as f32 * 0.95) as usize;
    let low_percentile = sorted_loudnesses[low_idx.min(sorted_loudnesses.len() - 1)];
    let high_percentile = sorted_loudnesses[high_idx.min(sorted_loudnesses.len() - 1)];

    let loudness_range = high_percentile - low_percentile;

    (loudness_range, window_loudnesses)
}

/// Calculate momentary loudness values (400ms windows)
fn calculate_momentary_loudness(weighted: &[f32], sample_rate: f32) -> Vec<f32> {
    let window_size = (sample_rate * 0.4) as usize; // 400ms windows
    let hop_size = (sample_rate * 0.1) as usize; // 100ms hop

    if weighted.len() < window_size {
        return vec![];
    }

    let mut momentary_loudnesses = Vec::new();
    let mut window_start = 0;

    while window_start + window_size <= weighted.len() {
        let window_end = window_start + window_size;
        let window_samples = &weighted[window_start..window_end];

        // Calculate mean square for this window
        let window_mean_square = window_samples
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / window_samples.len() as f64;

        // Convert to LUFS
        let window_loudness = if window_mean_square > 1e-10 {
            -0.691 + 10.0 * window_mean_square.log10()
        } else {
            -70.0
        };

        momentary_loudnesses.push(window_loudness as f32);
        window_start += hop_size;
    }

    momentary_loudnesses
}

/// Compute dynamics trajectory features from short-term loudness values.
///
/// Returns (entropy, slope_per_minute, peak_count).
fn compute_dynamics_trajectory(short_term_loudness: &[f32]) -> (f32, f32, u32) {
    if short_term_loudness.len() < 2 {
        return (0.0, 0.0, 0);
    }

    // --- Dynamics entropy: Shannon entropy of 1 dB LUFS histogram ---
    let min_lufs = short_term_loudness.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_lufs = short_term_loudness.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_lufs - min_lufs;

    let entropy = if range > 0.5 {
        let num_bins = (range.ceil() as usize).max(2); // 1 dB per bin
        let mut hist = vec![0u32; num_bins];
        for &v in short_term_loudness {
            let bin = ((v - min_lufs) as usize).min(num_bins - 1);
            hist[bin] += 1;
        }
        let n = short_term_loudness.len() as f32;
        let mut h = 0.0_f32;
        for &count in &hist {
            if count > 0 {
                let p = count as f32 / n;
                h -= p * p.log2();
            }
        }
        // Normalize by log2(num_bins) to get 0-1 range
        let max_entropy = (num_bins as f32).log2();
        if max_entropy > 0.0 { (h / max_entropy).clamp(0.0, 1.0) } else { 0.0 }
    } else {
        0.0 // Near-constant loudness
    };

    // --- Dynamics slope: least-squares LUFS/minute ---
    // short_term_loudness has 100ms hop → each index = 0.1s
    let n = short_term_loudness.len() as f64;
    let hop_seconds = 0.1_f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;

    for (i, &v) in short_term_loudness.iter().enumerate() {
        let x = i as f64 * hop_seconds; // time in seconds
        let y = v as f64;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let denom = n * sum_xx - sum_x * sum_x;
    let slope_per_second = if denom.abs() > 1e-10 {
        (n * sum_xy - sum_x * sum_y) / denom
    } else {
        0.0
    };
    let slope_per_minute = (slope_per_second * 60.0) as f32; // LUFS/minute

    // --- Dynamics peak count: local maxima with ≥3 LU prominence ---
    let mut peak_count = 0_u32;
    for i in 1..short_term_loudness.len() - 1 {
        let v = short_term_loudness[i];
        if v > short_term_loudness[i - 1] && v > short_term_loudness[i + 1] {
            // Find prominence: scan left and right for minimum
            let left_min = short_term_loudness[..i].iter().cloned().fold(v, f32::min);
            let right_min = short_term_loudness[i + 1..].iter().cloned().fold(v, f32::min);
            let prominence = v - left_min.max(right_min); // use the higher of the two valleys
            if prominence >= 3.0 {
                peak_count += 1;
            }
        }
    }

    (entropy, slope_per_minute, peak_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_metrics() {
        let samples = vec![0.0f32; 44100]; // 1 second of silence
        let metrics = calculate_perceptual_metrics(&samples, 1, 44100.0).unwrap();

        assert!(metrics.loudness_lufs < -60.0);
        assert_eq!(metrics.energy_level, 0.0);
        assert_eq!(metrics.dynamic_range, 0.0);
        assert_eq!(metrics.crest_factor, 0.0);
    }

    #[test]
    fn test_full_scale_sine() {
        // Generate a full-scale sine wave
        let mut samples = vec![0.0f32; 44100]; // 1 second
        for (i, sample) in samples.iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin();
        }

        let metrics = calculate_perceptual_metrics(&samples, 1, 44100.0).unwrap();

        // Full scale sine should be around -3 dBFS RMS
        assert!(metrics.true_peak_dbfs > -1.0 && metrics.true_peak_dbfs <= 0.0);
        assert!(metrics.crest_factor > 1.0); // Sine wave crest factor ≈ 1.414
        assert!(metrics.energy_level > 0.5); // Should have high energy
    }

    #[test]
    fn test_k_weighting_filter() {
        let filter = KWeightingFilter::new(48000.0);
        let samples = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let filtered = filter.apply(&samples);

        assert_eq!(filtered.len(), samples.len());
        // K-weighting should modify the signal
        assert_ne!(filtered[1], samples[1]);
    }

    #[test]
    fn test_dynamic_range_calculation() {
        // Create audio with varying dynamics
        let mut samples = vec![0.0f32; 176400]; // 4 seconds at 44.1kHz (need 3+ seconds for loudness range)

        // First second: quiet (0.1 amplitude)
        for (i, sample) in samples.iter_mut().enumerate().take(44100) {
            *sample = 0.1 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        }

        // Second second: medium (0.5 amplitude)
        for (i, sample) in samples.iter_mut().enumerate().skip(44100).take(44100) {
            *sample = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        }

        // Third second: loud (0.9 amplitude)
        for (i, sample) in samples.iter_mut().enumerate().skip(88200).take(44100) {
            *sample = 0.9 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        }

        // Fourth second: quiet again (0.2 amplitude)
        for (i, sample) in samples.iter_mut().enumerate().skip(132300).take(44100) {
            *sample = 0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        }

        let metrics = calculate_perceptual_metrics(&samples, 1, 44100.0).unwrap();

        // Should have significant dynamic range
        assert!(metrics.dynamic_range > 2.0);
        assert!(metrics.loudness_range > 0.0);
    }

    #[test]
    fn test_stereo_to_mono_conversion() {
        // Create stereo audio with different channels
        let mut samples = vec![0.0f32; 88200]; // 44100 frames * 2 channels

        // Left channel: 0.5 amplitude sine
        // Right channel: 0.3 amplitude sine
        for i in 0..44100 {
            let sine = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
            samples[i * 2] = 0.5 * sine;
            samples[i * 2 + 1] = 0.3 * sine;
        }

        let metrics = calculate_perceptual_metrics(&samples, 2, 44100.0).unwrap();

        // Should process successfully and return valid metrics
        assert!(metrics.loudness_lufs > -70.0);
        assert!(metrics.true_peak_dbfs > -20.0);
        assert!(metrics.crest_factor > 0.0);
    }
}
