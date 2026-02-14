use super::{normalized_square_difference, PitchDetector, PitchResult};
use std::collections::BTreeMap;

pub struct PyinDetector {
    num_candidates: usize,
    beta_parameters: (f32, f32),
    window_size: usize,
    min_frequency: f32,
    max_frequency: f32,
    threshold_distribution: Vec<f32>,
}

impl PyinDetector {
    pub fn new() -> Self {
        Self {
            num_candidates: 10,
            beta_parameters: (10.0, 1.0),
            window_size: 2048,
            min_frequency: 50.0,
            max_frequency: 2000.0,
            threshold_distribution: Self::generate_threshold_distribution_n(100),
        }
    }

    /// Set the number of PYIN thresholds (default 100, lower = faster).
    /// 25 gives good results with ~4x speedup in the threshold sweep.
    pub fn with_threshold_count(mut self, count: usize) -> Self {
        let count = count.clamp(5, 200);
        self.threshold_distribution = Self::generate_threshold_distribution_n(count);
        self
    }

    pub fn with_candidates(mut self, num: usize) -> Self {
        self.num_candidates = num.clamp(3, 20);
        self
    }

    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    pub fn with_frequency_range(mut self, min: f32, max: f32) -> Self {
        self.min_frequency = min;
        self.max_frequency = max;
        self
    }

    fn generate_threshold_distribution_n(num_thresholds: usize) -> Vec<f32> {
        let mut thresholds = Vec::with_capacity(num_thresholds);

        for i in 0..num_thresholds {
            let t = 0.01 + (i as f32 / num_thresholds as f32) * 0.49;
            thresholds.push(t);
        }

        thresholds
    }

    fn difference_function(&self, samples: &[f32]) -> Vec<f32> {
        let w = samples.len();
        let half_w = w / 2;

        (0..half_w)
            .map(|tau| {
                if tau == 0 {
                    0.0
                } else {
                    normalized_square_difference(samples, tau)
                }
            })
            .collect()
    }

    fn cumulative_mean_normalized_difference(&self, diff: &[f32]) -> Vec<f32> {
        let mut cmnd = vec![0.0; diff.len()];
        cmnd[0] = 1.0;

        let mut running_sum = 0.0;
        for tau in 1..diff.len() {
            running_sum += diff[tau];
            if running_sum > 0.0 {
                cmnd[tau] = diff[tau] * tau as f32 / running_sum;
            } else {
                cmnd[tau] = 1.0;
            }
        }

        cmnd
    }

    fn get_pitch_candidates(&self, cmnd: &[f32], sample_rate: f32) -> Vec<(f32, f32)> {
        let mut candidates = BTreeMap::new();

        for &threshold in &self.threshold_distribution {
            // Find the first significant local minimum for this threshold
            let mut best_tau = None;

            for tau in 2..cmnd.len() - 1 {
                if cmnd[tau] < threshold && cmnd[tau] < cmnd[tau - 1] && cmnd[tau] < cmnd[tau + 1] {
                    let min_tau = (sample_rate / self.max_frequency) as usize;
                    let max_tau = (sample_rate / self.min_frequency) as usize;

                    if tau >= min_tau && tau <= max_tau {
                        best_tau = Some(tau);
                        break; // Take the first good minimum for each threshold
                    }
                }
            }

            if let Some(tau) = best_tau {
                let refined_tau = self.parabolic_interpolation(cmnd, tau);
                let frequency = sample_rate / refined_tau;
                let weight = self.calculate_weight(cmnd[tau], threshold);

                // Aggregate by frequency bins
                let bin = (frequency / 5.0).round() as i32;
                candidates
                    .entry(bin)
                    .and_modify(|e| *e += weight)
                    .or_insert(weight);
            }
        }

        let mut sorted_candidates: Vec<_> = candidates
            .into_iter()
            .map(|(bin, w)| (bin as f32 * 5.0, w))
            .collect();

        sorted_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted_candidates.truncate(self.num_candidates);
        sorted_candidates
    }

    fn parabolic_interpolation(&self, cmnd: &[f32], tau: usize) -> f32 {
        if tau == 0 || tau >= cmnd.len() - 1 {
            return tau as f32;
        }

        let y0 = cmnd[tau - 1];
        let y1 = cmnd[tau];
        let y2 = cmnd[tau + 1];

        let a = (y2 - 2.0 * y1 + y0) / 2.0;
        let b = (y2 - y0) / 2.0;

        if a.abs() < f32::EPSILON {
            return tau as f32;
        }

        let x_vertex = tau as f32 - b / (2.0 * a);
        x_vertex.max(1.0)
    }

    fn calculate_weight(&self, cmnd_value: f32, threshold: f32) -> f32 {
        let alpha = self.beta_parameters.0;
        let beta = self.beta_parameters.1;

        let x = (1.0 - cmnd_value).clamp(0.0, 1.0);
        let threshold_weight = (-10.0 * (threshold - 0.15).abs()).exp();

        x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0) * threshold_weight
    }
}

impl Default for PyinDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PitchDetector for PyinDetector {
    fn detect_pitch(&self, samples: &[f32], sample_rate: f32) -> PitchResult {
        if samples.is_empty() {
            return PitchResult::new(0.0, 0.0, 0.0);
        }

        let effective_window = samples.len().min(self.window_size);
        let windowed = &samples[..effective_window];

        let diff = self.difference_function(windowed);
        let cmnd = self.cumulative_mean_normalized_difference(&diff);

        let candidates = self.get_pitch_candidates(&cmnd, sample_rate);

        if candidates.is_empty() {
            return PitchResult::new(0.0, 0.0, 0.0);
        }

        let (best_freq, confidence) = candidates[0];

        let total_weight: f32 = candidates.iter().map(|(_, w)| w).sum();
        let normalized_confidence = if total_weight > 0.0 {
            confidence / total_weight
        } else {
            0.0
        };

        let clarity = if candidates.len() > 1 {
            let ratio = confidence / candidates[1].1.max(f32::EPSILON);
            (ratio - 1.0) / (ratio + 1.0)
        } else {
            1.0
        };

        PitchResult::new(best_freq, normalized_confidence, clarity)
    }

    fn window_size(&self) -> usize {
        self.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_sine_wave(frequency: f32, sample_rate: f32, duration: f32) -> Vec<f32> {
        let num_samples = (sample_rate * duration) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (2.0 * PI * frequency * t).sin()
            })
            .collect()
    }

    fn generate_complex_tone(
        fundamental: f32,
        harmonics: &[(f32, f32)],
        sample_rate: f32,
        duration: f32,
    ) -> Vec<f32> {
        let num_samples = (sample_rate * duration) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let mut sample = (2.0 * PI * fundamental * t).sin();

                for &(harmonic_ratio, amplitude) in harmonics {
                    sample += amplitude * (2.0 * PI * fundamental * harmonic_ratio * t).sin();
                }

                sample / (1.0 + harmonics.len() as f32)
            })
            .collect()
    }

    #[test]
    fn test_pyin_pure_sine() {
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let samples = generate_sine_wave(frequency, sample_rate, 0.1);

        let detector = PyinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        assert!((result.frequency - frequency).abs() < 5.0);
        assert!(result.confidence > 0.7);
    }

    #[test]
    fn test_pyin_complex_tone() {
        let sample_rate = 44100.0;
        let fundamental = 220.0;
        let harmonics = vec![(2.0, 0.5), (3.0, 0.3), (4.0, 0.2)];
        let samples = generate_complex_tone(fundamental, &harmonics, sample_rate, 0.1);

        let detector = PyinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        assert!((result.frequency - fundamental).abs() < 10.0);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_pyin_with_vibrato() {
        let sample_rate = 44100.0;
        let base_freq = 440.0;
        let vibrato_freq = 5.0;
        let vibrato_depth = 0.02;

        let num_samples = (sample_rate * 0.2) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                let freq = base_freq * (1.0 + vibrato_depth * (2.0 * PI * vibrato_freq * t).sin());
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let detector = PyinDetector::new();
        let result = detector.detect_pitch(&samples, sample_rate);

        assert!((result.frequency - base_freq).abs() < 20.0);
    }
}
