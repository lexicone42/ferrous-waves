pub struct BeatTracker {
    min_tempo: f32,
    max_tempo: f32,
}

impl BeatTracker {
    pub fn new() -> Self {
        Self {
            min_tempo: 50.0,
            max_tempo: 220.0,
        }
    }

    /// Estimate tempo from an onset strength envelope using autocorrelation.
    ///
    /// The onset envelope (spectral flux) is autocorrelated at lags corresponding
    /// to the valid tempo range (50-220 BPM). The lag with the highest
    /// autocorrelation is the estimated beat period.
    ///
    /// This replaces the previous histogram-of-IOIs approach which produced
    /// only ~28 distinct BPM values due to 10ms bin quantization.
    pub fn estimate_tempo_from_envelope(
        &self,
        onset_envelope: &[f32],
        hop_size: usize,
        sample_rate: u32,
    ) -> Option<f32> {
        if onset_envelope.len() < 64 {
            return None;
        }

        let frame_duration = hop_size as f32 / sample_rate as f32;

        // Lag range in frames for valid tempo range
        let min_lag = (60.0 / (self.max_tempo * frame_duration)).floor() as usize;
        let max_lag = (60.0 / (self.min_tempo * frame_duration)).ceil() as usize;
        let max_lag = max_lag.min(onset_envelope.len() / 2);

        if min_lag >= max_lag || max_lag >= onset_envelope.len() {
            return None;
        }

        // Subtract mean to remove DC bias
        let mean = onset_envelope.iter().sum::<f32>() / onset_envelope.len() as f32;
        let centered: Vec<f32> = onset_envelope.iter().map(|&x| x - mean).collect();

        // Autocorrelation for each candidate lag
        let n = centered.len();
        let mut best_lag = min_lag;
        let mut best_corr = f32::NEG_INFINITY;

        // Normalization: autocorrelation at lag 0
        let energy: f32 = centered.iter().map(|&x| x * x).sum();
        if energy < 1e-10 {
            return None;
        }

        for lag in min_lag..=max_lag {
            let corr: f32 = centered[..n - lag]
                .iter()
                .zip(centered[lag..].iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                / energy;

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        // Require minimum correlation strength
        if best_corr < 0.05 {
            return None;
        }

        // Parabolic interpolation around the peak for sub-frame precision
        let tempo_lag = if best_lag > min_lag && best_lag < max_lag {
            let corr_at = |lag: usize| -> f32 {
                centered[..n - lag]
                    .iter()
                    .zip(centered[lag..].iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>()
                    / energy
            };

            let prev = corr_at(best_lag - 1);
            let curr = best_corr;
            let next = corr_at(best_lag + 1);

            // Parabolic fit: offset = 0.5 * (prev - next) / (prev - 2*curr + next)
            let denom = prev - 2.0 * curr + next;
            if denom.abs() > 1e-10 {
                best_lag as f32 + 0.5 * (prev - next) / denom
            } else {
                best_lag as f32
            }
        } else {
            best_lag as f32
        };

        let beat_period = tempo_lag * frame_duration;
        if beat_period > 0.0 {
            let bpm = 60.0 / beat_period;
            // Handle octave ambiguity: if BPM is very high, check half-tempo
            if bpm > 160.0 {
                let half_lag = (tempo_lag * 2.0).round() as usize;
                if half_lag <= max_lag {
                    let half_corr: f32 = centered[..n - half_lag]
                        .iter()
                        .zip(centered[half_lag..].iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                        / energy;
                    // Prefer half-tempo if its correlation is at least 60% of the peak.
                    // Jam-band tracks at 160+ BPM often have half-period correlations
                    // in the 0.6-0.75 range due to polyrhythmic complexity.
                    if half_corr > best_corr * 0.6 {
                        return Some(bpm / 2.0);
                    }
                }
            }
            Some(bpm)
        } else {
            None
        }
    }

    /// Legacy tempo estimation from onset times (IOI histogram).
    /// Kept for backward compatibility but estimate_tempo_from_envelope is preferred.
    pub fn estimate_tempo(&self, onset_times: &[f32]) -> Option<f32> {
        if onset_times.len() < 4 {
            return None;
        }

        // Compute inter-onset intervals
        let intervals: Vec<f32> = onset_times.windows(2).map(|w| w[1] - w[0]).collect();

        // Use kernel density estimation instead of fixed bins
        let min_interval = 60.0 / self.max_tempo;
        let max_interval = 60.0 / self.min_tempo;

        // Filter to valid tempo range
        let valid: Vec<f32> = intervals
            .iter()
            .copied()
            .filter(|&i| i >= min_interval && i <= max_interval)
            .collect();

        if valid.is_empty() {
            return None;
        }

        // Weighted median — more robust than mode with quantized bins
        let mut sorted = valid.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = sorted[sorted.len() / 2];
        let bpm = 60.0 / median;

        // Handle octave ambiguity
        if bpm > 160.0 {
            // Check if half-tempo intervals are also well-represented
            let half_min = median * 1.8;
            let half_max = median * 2.2;
            let half_count = intervals
                .iter()
                .filter(|&&i| i >= half_min && i <= half_max)
                .count();
            let base_count = valid.len();
            if half_count as f32 > base_count as f32 * 0.3 {
                return Some(bpm / 2.0);
            }
        }

        Some(bpm)
    }

    pub fn track_beats(&self, onset_times: &[f32], tempo: f32) -> Vec<f32> {
        if onset_times.is_empty() {
            return Vec::new();
        }

        let beat_period = 60.0 / tempo;
        let mut beats = Vec::new();

        // Find the best phase offset
        let mut best_phase = 0.0;
        let mut best_score = 0.0;

        for &onset in onset_times.iter().take(10) {
            let mut score = 0.0;
            let mut beat_time = onset;

            while beat_time < onset_times[onset_times.len() - 1] {
                // Find closest onset
                for &o in onset_times {
                    let distance = (o - beat_time).abs();
                    if distance < beat_period * 0.2 {
                        score += 1.0 / (1.0 + distance);
                    }
                }
                beat_time += beat_period;
            }

            if score > best_score {
                best_score = score;
                best_phase = onset;
            }
        }

        // Generate beats
        let mut beat_time = best_phase;
        let duration = onset_times[onset_times.len() - 1];

        while beat_time <= duration {
            beats.push(beat_time);
            beat_time += beat_period;
        }

        beats
    }

    /// Estimate tempo in overlapping windows across the track.
    ///
    /// Returns `(bpm, peak_correlation)` pairs for each window. The correlation
    /// strength measures beat clarity (high = strong beat, low = noise/free time).
    /// Windows that fail to find any periodicity are omitted.
    pub fn estimate_tempo_windowed(
        &self,
        onset_envelope: &[f32],
        hop_size: usize,
        sample_rate: u32,
        window_frames: usize,
        hop_frames: usize,
    ) -> Vec<(f32, f32)> {
        let mut results = Vec::new();
        let mut start = 0;

        let frame_duration = hop_size as f32 / sample_rate as f32;
        let min_lag = (60.0 / (self.max_tempo * frame_duration)).floor() as usize;
        let max_lag_limit = (60.0 / (self.min_tempo * frame_duration)).ceil() as usize;

        while start + window_frames <= onset_envelope.len() {
            let window = &onset_envelope[start..start + window_frames];

            if window.len() >= 64 {
                let max_lag = max_lag_limit.min(window.len() / 2);
                if min_lag < max_lag && max_lag < window.len() {
                    let mean = window.iter().sum::<f32>() / window.len() as f32;
                    let centered: Vec<f32> = window.iter().map(|&x| x - mean).collect();
                    let energy: f32 = centered.iter().map(|&x| x * x).sum();

                    if energy > 1e-10 {
                        let n = centered.len();
                        let mut best_lag = min_lag;
                        let mut best_corr = f32::NEG_INFINITY;

                        for lag in min_lag..=max_lag {
                            let corr: f32 = centered[..n - lag]
                                .iter()
                                .zip(centered[lag..].iter())
                                .map(|(&a, &b)| a * b)
                                .sum::<f32>()
                                / energy;
                            if corr > best_corr {
                                best_corr = corr;
                                best_lag = lag;
                            }
                        }

                        // Minimal threshold: only reject windows with no periodicity at all
                        if best_corr >= 0.005 {
                            let tempo_lag = if best_lag > min_lag && best_lag < max_lag {
                                let corr_at = |lag: usize| -> f32 {
                                    centered[..n - lag]
                                        .iter()
                                        .zip(centered[lag..].iter())
                                        .map(|(&a, &b)| a * b)
                                        .sum::<f32>()
                                        / energy
                                };
                                let prev = corr_at(best_lag - 1);
                                let curr = best_corr;
                                let next = corr_at(best_lag + 1);
                                let denom = prev - 2.0 * curr + next;
                                if denom.abs() > 1e-10 {
                                    best_lag as f32 + 0.5 * (prev - next) / denom
                                } else {
                                    best_lag as f32
                                }
                            } else {
                                best_lag as f32
                            };

                            let beat_period = tempo_lag * frame_duration;
                            if beat_period > 0.0 {
                                let bpm = 60.0 / beat_period;
                                results.push((bpm, best_corr));
                            }
                        }
                    }
                }
            }

            start += hop_frames;
        }

        results
    }
}

impl Default for BeatTracker {
    fn default() -> Self {
        Self::new()
    }
}
