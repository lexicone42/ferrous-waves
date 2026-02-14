use crate::analysis::spectral::StftProcessor;
use crate::utils::error::Result;
use ndarray::Array2;

/// Classification of audio content type
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ContentType {
    Speech,
    Music,
    Silence,
    Mixed,
}

/// Detailed classification results with confidence scores
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContentClassification {
    /// Primary content type
    pub primary_type: ContentType,

    /// Confidence score for the primary type (0.0 to 1.0)
    pub confidence: f32,

    /// Individual confidence scores for each type
    pub scores: ClassificationScores,

    /// Segment-based classification for temporal analysis
    pub segments: Vec<SegmentClassification>,

    /// Additional features used for classification
    pub features: ClassificationFeatures,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassificationScores {
    pub speech: f32,
    pub music: f32,
    pub silence: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentClassification {
    pub start_time: f32,
    pub end_time: f32,
    pub content_type: ContentType,
    pub confidence: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassificationFeatures {
    /// Zero crossing rate - higher for speech
    pub zcr_mean: f32,
    pub zcr_std: f32,

    /// Spectral rolloff - lower for speech
    pub spectral_rolloff_mean: f32,

    /// Spectral centroid variance - lower for speech
    pub spectral_centroid_variance: f32,

    /// Spectral flux - higher for music
    pub spectral_flux_mean: f32,

    /// RMS energy
    pub energy_mean: f32,
    pub energy_std: f32,

    /// Low energy rate - higher for speech
    pub low_energy_rate: f32,

    /// Harmonic to noise ratio
    pub hnr: f32,
}

/// Content classifier using spectral and temporal features
pub struct ContentClassifier {
    frame_size: usize,
    hop_size: usize,
    sample_rate: f32,
    silence_threshold: f32,
}

impl ContentClassifier {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            frame_size: 2048,
            hop_size: 512,
            sample_rate,
            silence_threshold: 0.01, // RMS threshold for silence
        }
    }

    pub fn classify(&self, samples: &[f32]) -> Result<ContentClassification> {
        self.classify_with_options(samples, None, false)
    }

    /// Classify with optional pre-computed spectrogram and option to skip per-segment classification.
    pub fn classify_with_options(
        &self,
        samples: &[f32],
        precomputed_spectrogram: Option<&Array2<f32>>,
        skip_segments: bool,
    ) -> Result<ContentClassification> {
        // Calculate features (reuse spectrogram if provided)
        let features = self.extract_features_with_spectrogram(samples, precomputed_spectrogram)?;

        // Classify based on features
        let scores = self.calculate_scores(&features);

        // Determine primary type and confidence
        let (primary_type, confidence) = self.determine_primary_type(&scores);

        // Perform segment-based classification (expensive — skip if configured)
        let segments = if skip_segments {
            Vec::new()
        } else {
            self.classify_segments(samples)?
        };

        Ok(ContentClassification {
            primary_type,
            confidence,
            scores,
            segments,
            features,
        })
    }

    fn extract_features(&self, samples: &[f32]) -> Result<ClassificationFeatures> {
        self.extract_features_with_spectrogram(samples, None)
    }

    fn extract_features_with_spectrogram(
        &self,
        samples: &[f32],
        precomputed_spectrogram: Option<&Array2<f32>>,
    ) -> Result<ClassificationFeatures> {
        // Calculate zero crossing rate
        let zcr_values = self.calculate_zcr(samples);
        let zcr_mean = zcr_values.iter().sum::<f32>() / zcr_values.len() as f32;
        let zcr_variance = zcr_values
            .iter()
            .map(|&x| (x - zcr_mean).powi(2))
            .sum::<f32>()
            / zcr_values.len() as f32;
        let zcr_std = zcr_variance.sqrt();

        // Calculate RMS energy
        let energy_values = self.calculate_energy(samples);
        let energy_mean = energy_values.iter().sum::<f32>() / energy_values.len() as f32;
        let energy_variance = energy_values
            .iter()
            .map(|&x| (x - energy_mean).powi(2))
            .sum::<f32>()
            / energy_values.len() as f32;
        let energy_std = energy_variance.sqrt();

        // Calculate low energy rate
        let low_energy_threshold = energy_mean * 0.5;
        let low_energy_count = energy_values
            .iter()
            .filter(|&&x| x < low_energy_threshold)
            .count();
        let low_energy_rate = low_energy_count as f32 / energy_values.len() as f32;

        // Reuse pre-computed spectrogram if available (same 2048/512 params)
        let owned_spectrogram;
        let spectrogram = match precomputed_spectrogram {
            Some(sg) => sg,
            None => {
                let stft_processor = StftProcessor::new(
                    self.frame_size,
                    self.hop_size,
                    crate::analysis::spectral::WindowFunction::Hann,
                );
                owned_spectrogram = stft_processor.process(samples);
                &owned_spectrogram
            }
        };

        // Calculate spectral features
        let (spectral_rolloff_mean, spectral_centroid_variance, spectral_flux_mean) =
            self.calculate_spectral_features(spectrogram);

        // Calculate harmonic to noise ratio (simplified)
        let hnr = self.calculate_hnr(samples);

        Ok(ClassificationFeatures {
            zcr_mean,
            zcr_std,
            spectral_rolloff_mean,
            spectral_centroid_variance,
            spectral_flux_mean,
            energy_mean,
            energy_std,
            low_energy_rate,
            hnr,
        })
    }

    fn calculate_zcr(&self, samples: &[f32]) -> Vec<f32> {
        let mut zcr_values = Vec::new();
        let frame_size = self.frame_size;
        let hop_size = self.hop_size;

        let mut pos = 0;
        while pos + frame_size <= samples.len() {
            let frame = &samples[pos..pos + frame_size];

            let mut zcr_count = 0;
            for i in 1..frame.len() {
                if (frame[i] >= 0.0) != (frame[i - 1] >= 0.0) {
                    zcr_count += 1;
                }
            }

            let zcr = zcr_count as f32 / frame_size as f32;
            zcr_values.push(zcr);

            pos += hop_size;
        }

        zcr_values
    }

    fn calculate_energy(&self, samples: &[f32]) -> Vec<f32> {
        let mut energy_values = Vec::new();
        let frame_size = self.frame_size;
        let hop_size = self.hop_size;

        let mut pos = 0;
        while pos + frame_size <= samples.len() {
            let frame = &samples[pos..pos + frame_size];

            let energy = frame.iter().map(|&x| x * x).sum::<f32>() / frame_size as f32;
            let rms = energy.sqrt();
            energy_values.push(rms);

            pos += hop_size;
        }

        energy_values
    }

    fn calculate_spectral_features(&self, spectrogram: &Array2<f32>) -> (f32, f32, f32) {
        let num_frames = spectrogram.shape()[1];
        let num_bins = spectrogram.shape()[0];

        let mut rolloff_values = Vec::new();
        let mut centroid_values = Vec::new();
        let mut flux_values = Vec::new();

        for frame_idx in 0..num_frames {
            let frame = spectrogram.column(frame_idx);

            // Spectral rolloff (85% of energy)
            let total_energy: f32 = frame.iter().sum();
            let rolloff_threshold = total_energy * 0.85;
            let mut cumulative_energy = 0.0;
            let mut rolloff_bin = 0;

            for (bin, &mag) in frame.iter().enumerate() {
                cumulative_energy += mag;
                if cumulative_energy >= rolloff_threshold {
                    rolloff_bin = bin;
                    break;
                }
            }

            let rolloff_freq = rolloff_bin as f32 * self.sample_rate / (2.0 * num_bins as f32);
            rolloff_values.push(rolloff_freq);

            // Spectral centroid
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;

            for (bin, &mag) in frame.iter().enumerate() {
                let freq = bin as f32 * self.sample_rate / (2.0 * num_bins as f32);
                weighted_sum += freq * mag;
                magnitude_sum += mag;
            }

            if magnitude_sum > 0.0 {
                centroid_values.push(weighted_sum / magnitude_sum);
            }

            // Spectral flux
            if frame_idx > 0 {
                let prev_frame = spectrogram.column(frame_idx - 1);
                let flux: f32 = frame
                    .iter()
                    .zip(prev_frame.iter())
                    .map(|(&curr, &prev)| (curr - prev).max(0.0).powi(2))
                    .sum();
                flux_values.push(flux.sqrt());
            }
        }

        // Calculate means and variance
        let rolloff_mean = rolloff_values.iter().sum::<f32>() / rolloff_values.len() as f32;

        let centroid_mean = centroid_values.iter().sum::<f32>() / centroid_values.len() as f32;
        let centroid_variance = centroid_values
            .iter()
            .map(|&x| (x - centroid_mean).powi(2))
            .sum::<f32>()
            / centroid_values.len() as f32;

        let flux_mean = if !flux_values.is_empty() {
            flux_values.iter().sum::<f32>() / flux_values.len() as f32
        } else {
            0.0
        };

        (rolloff_mean, centroid_variance, flux_mean)
    }

    fn calculate_hnr(&self, samples: &[f32]) -> f32 {
        // Simplified HNR calculation using autocorrelation
        // Higher values indicate more harmonic content (typical of speech)

        let frame_size = (0.04 * self.sample_rate) as usize; // 40ms frame
        let mut hnr_values = Vec::new();

        let mut pos = 0;
        while pos + frame_size <= samples.len() {
            let frame = &samples[pos..pos + frame_size];

            // Calculate autocorrelation for pitch detection
            let mut max_correlation = 0.0f32;
            let min_period = (self.sample_rate / 500.0) as usize; // 500 Hz max
            let max_period = (self.sample_rate / 50.0) as usize; // 50 Hz min

            for lag in min_period..max_period.min(frame_size / 2) {
                let mut correlation = 0.0;
                let mut energy = 0.0;

                for i in 0..frame_size - lag {
                    correlation += frame[i] * frame[i + lag];
                    energy += frame[i] * frame[i] + frame[i + lag] * frame[i + lag];
                }

                if energy > 0.0 {
                    let normalized = 2.0 * correlation / energy;
                    max_correlation = max_correlation.max(normalized);
                }
            }

            hnr_values.push(max_correlation);
            pos += frame_size / 2;
        }

        if !hnr_values.is_empty() {
            hnr_values.iter().sum::<f32>() / hnr_values.len() as f32
        } else {
            0.0
        }
    }

    fn calculate_scores(&self, features: &ClassificationFeatures) -> ClassificationScores {
        // Silence detection based on energy
        let silence_score = if features.energy_mean < self.silence_threshold {
            0.95
        } else {
            0.05
        };

        if silence_score > 0.5 {
            return ClassificationScores {
                speech: 0.02,
                music: 0.03,
                silence: silence_score,
            };
        }

        // Speech vs Music classification based on features
        let mut speech_score = 0.0;
        let mut music_score = 0.0;

        // Zero crossing rate - higher and more variable for speech
        if features.zcr_mean > 0.05 && features.zcr_std > 0.02 {
            speech_score += 0.2;
        } else {
            music_score += 0.2;
        }

        // Spectral rolloff - lower for speech (most energy in lower frequencies)
        if features.spectral_rolloff_mean < 2000.0 {
            speech_score += 0.15;
        } else if features.spectral_rolloff_mean > 4000.0 {
            music_score += 0.15;
        }

        // Spectral centroid variance - lower for speech (more stable pitch)
        if features.spectral_centroid_variance < 1000000.0 {
            speech_score += 0.15;
        } else {
            music_score += 0.15;
        }

        // Low energy rate - higher for speech (pauses between words)
        if features.low_energy_rate > 0.3 {
            speech_score += 0.2;
        } else {
            music_score += 0.2;
        }

        // HNR - higher for speech (harmonic structure of voice)
        if features.hnr > 0.3 {
            speech_score += 0.15;
        } else {
            music_score += 0.1;
        }

        // Energy variation - moderate for speech
        let energy_cv = features.energy_std / features.energy_mean.max(0.001);
        if energy_cv > 0.3 && energy_cv < 0.8 {
            speech_score += 0.1;
        } else {
            music_score += 0.1;
        }

        // Spectral flux - higher for music (more variation)
        if features.spectral_flux_mean > 0.5 {
            music_score += 0.15;
        } else {
            speech_score += 0.1;
        }

        // Normalize scores
        let total = speech_score + music_score + silence_score;

        ClassificationScores {
            speech: speech_score / total,
            music: music_score / total,
            silence: silence_score / total,
        }
    }

    fn determine_primary_type(&self, scores: &ClassificationScores) -> (ContentType, f32) {
        if scores.silence > scores.speech && scores.silence > scores.music {
            (ContentType::Silence, scores.silence)
        } else if scores.speech > scores.music {
            if scores.speech > 0.7 {
                (ContentType::Speech, scores.speech)
            } else if scores.music > 0.3 {
                (ContentType::Mixed, scores.speech.max(scores.music))
            } else {
                (ContentType::Speech, scores.speech)
            }
        } else if scores.music > 0.7 {
            (ContentType::Music, scores.music)
        } else if scores.speech > 0.3 {
            (ContentType::Mixed, scores.speech.max(scores.music))
        } else {
            (ContentType::Music, scores.music)
        }
    }

    fn classify_segments(&self, samples: &[f32]) -> Result<Vec<SegmentClassification>> {
        let segment_duration = 1.0; // 1 second segments
        let segment_samples = (segment_duration * self.sample_rate) as usize;
        let hop_samples = segment_samples / 2; // 50% overlap

        let mut segments = Vec::new();
        let mut pos = 0;
        let mut segment_idx = 0;

        while pos + segment_samples <= samples.len() {
            let segment = &samples[pos..pos + segment_samples];

            // Extract features for this segment
            let features = self.extract_features(segment)?;
            let scores = self.calculate_scores(&features);
            let (content_type, confidence) = self.determine_primary_type(&scores);

            let start_time = segment_idx as f32 * segment_duration * 0.5; // Account for overlap
            let end_time = start_time + segment_duration;

            segments.push(SegmentClassification {
                start_time,
                end_time,
                content_type,
                confidence,
            });

            pos += hop_samples;
            segment_idx += 1;
        }

        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence_detection() {
        let samples = vec![0.0f32; 44100]; // 1 second of silence
        let classifier = ContentClassifier::new(44100.0);
        let result = classifier.classify(&samples).unwrap();

        assert_eq!(result.primary_type, ContentType::Silence);
        assert!(result.confidence > 0.9);
        assert!(result.scores.silence > 0.9);
    }

    #[test]
    fn test_pure_tone_classification() {
        // Generate a pure tone (more like music than speech)
        let mut samples = vec![0.0f32; 44100];
        for (i, sample) in samples.iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5;
        }

        let classifier = ContentClassifier::new(44100.0);
        let result = classifier.classify(&samples).unwrap();

        // Pure tone should not be classified as speech or silence
        assert_ne!(result.primary_type, ContentType::Speech);
        assert_ne!(result.primary_type, ContentType::Silence);
        // It will likely be Music or Mixed for a simple tone
        assert!(
            result.primary_type == ContentType::Music || result.primary_type == ContentType::Mixed
        );
    }

    #[test]
    fn test_feature_extraction() {
        let samples = vec![0.5f32; 44100];
        let classifier = ContentClassifier::new(44100.0);
        let features = classifier.extract_features(&samples).unwrap();

        // Check that features are calculated
        assert!(features.energy_mean > 0.0);
        assert_eq!(features.zcr_mean, 0.0); // Constant signal has no zero crossings
        assert!(features.hnr >= 0.0 && features.hnr <= 1.0);
    }

    #[test]
    fn test_segment_classification() {
        // Create 3 seconds: silence, tone, silence
        let mut samples = vec![0.0f32; 44100 * 3];

        // Middle second: tone
        for (i, sample) in samples.iter_mut().enumerate().skip(44100).take(44100) {
            *sample = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
        }

        let classifier = ContentClassifier::new(44100.0);
        let result = classifier.classify(&samples).unwrap();

        assert!(!result.segments.is_empty());
        // First segment should be silence
        assert_eq!(result.segments[0].content_type, ContentType::Silence);
    }
}
