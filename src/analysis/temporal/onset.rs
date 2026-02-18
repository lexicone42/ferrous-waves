use ndarray::{s, Array2};

pub struct OnsetDetector {
    /// Multiplicative threshold: a peak must exceed local_mean * threshold to be an onset.
    /// Default 1.5 means peak must be 50% above local average.
    threshold: f32,
    pre_max: usize,
    post_max: usize,
    pre_avg: usize,
    post_avg: usize,
}

impl OnsetDetector {
    pub fn new() -> Self {
        Self {
            threshold: 1.5,
            pre_max: 3,
            post_max: 3,
            pre_avg: 30,
            post_avg: 30,
        }
    }

    /// Compute normalized spectral flux from a spectrogram.
    /// Each flux value is normalized by frame energy so that amplitude/bit-depth
    /// does not affect the magnitude of detected changes.
    pub fn spectral_flux(&self, spectrogram: &Array2<f32>) -> Vec<f32> {
        let num_frames = spectrogram.shape()[1];
        let mut flux = vec![0.0; num_frames];

        for (i, flux_value) in flux.iter_mut().enumerate().skip(1).take(num_frames - 1) {
            let prev_frame = spectrogram.slice(s![.., i - 1]);
            let curr_frame = spectrogram.slice(s![.., i]);

            let raw: f32 = curr_frame
                .iter()
                .zip(prev_frame.iter())
                .map(|(&curr, &prev)| {
                    let d = curr - prev;
                    if d > 0.0 { d } else { 0.0 }
                })
                .sum();

            // Normalize by geometric mean of frame energies
            let curr_energy: f32 = curr_frame.iter().map(|&m| m * m).sum::<f32>();
            let prev_energy: f32 = prev_frame.iter().map(|&m| m * m).sum::<f32>();
            let norm = (curr_energy * prev_energy).sqrt().max(1e-10).sqrt();

            *flux_value = raw / norm;
        }

        flux
    }

    pub fn detect_onsets(
        &self,
        onset_function: &[f32],
        hop_size: usize,
        sample_rate: u32,
    ) -> Vec<f32> {
        let peaks = self.peak_pick(onset_function);

        // Convert frame indices to time
        peaks
            .iter()
            .map(|&idx| idx as f32 * hop_size as f32 / sample_rate as f32)
            .collect()
    }

    fn peak_pick(&self, signal: &[f32]) -> Vec<usize> {
        let mut peaks = Vec::new();
        let len = signal.len();

        for i in 0..len {
            // Check if local maximum
            let mut is_max = true;

            for j in i.saturating_sub(self.pre_max)..=(i + self.post_max).min(len - 1) {
                if signal[j] > signal[i] {
                    is_max = false;
                    break;
                }
            }

            if !is_max {
                continue;
            }

            // Compute adaptive threshold from local context
            let pre_start = i.saturating_sub(self.pre_avg);
            let pre_end = i.saturating_sub(1);
            let post_start = (i + 1).min(len - 1);
            let post_end = (i + self.post_avg).min(len - 1);

            let mut mean = 0.0;
            let mut count = 0;

            if pre_end >= pre_start {
                for &value in signal.iter().take(pre_end + 1).skip(pre_start) {
                    mean += value;
                    count += 1;
                }
            }

            if post_end >= post_start {
                for &value in signal.iter().take(post_end + 1).skip(post_start) {
                    mean += value;
                    count += 1;
                }
            }

            if count > 0 {
                mean /= count as f32;
            }

            // Multiplicative threshold: peak must exceed local mean by threshold ratio.
            // With normalized flux, this is amplitude-independent.
            if signal[i] > mean * self.threshold {
                peaks.push(i);
            }
        }

        peaks
    }
}

impl Default for OnsetDetector {
    fn default() -> Self {
        Self::new()
    }
}
