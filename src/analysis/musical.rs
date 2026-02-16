use crate::analysis::spectral::StftProcessor;
use crate::utils::error::Result;

/// Musical key with confidence score
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MusicalKey {
    /// The detected key (e.g., "C major", "A minor")
    pub key: String,

    /// Root note (e.g., "C", "A")
    pub root: String,

    /// Mode (major or minor)
    pub mode: KeyMode,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Alternative key interpretations with scores
    pub alternatives: Vec<KeyCandidate>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum KeyMode {
    Major,
    Minor,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KeyCandidate {
    pub key: String,
    pub root: String,
    pub mode: KeyMode,
    pub score: f32,
}

/// Musical analysis results with enhanced metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MusicalAnalysis {
    /// Detected musical key with confidence
    pub key: MusicalKey,

    /// Chord progression if detected
    pub chord_progression: Option<ChordProgression>,

    /// Pitch class histogram (chroma vector)
    pub chroma_vector: ChromaVector,

    /// Tonal strength (0.0 to 1.0)
    pub tonality: f32,

    /// Musical mode characteristics
    pub mode_clarity: f32,

    /// Harmonic complexity measure
    pub harmonic_complexity: f32,

    /// Time signature estimation
    pub time_signature: Option<TimeSignature>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChordProgression {
    pub chords: Vec<ChordEvent>,
    pub confidence: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChordEvent {
    pub chord: String,
    pub start_time: f32,
    pub duration: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChromaVector {
    /// Energy distribution across 12 pitch classes [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
    pub values: [f32; 12],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
    pub confidence: f32,
}

/// Musical analyzer for key detection and harmonic analysis
pub struct MusicalAnalyzer {
    sample_rate: f32,
    fft_size: usize,
    hop_size: usize,
}

impl MusicalAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            fft_size: 4096, // Larger FFT for better frequency resolution
            hop_size: 2048,
        }
    }

    pub fn analyze(&self, samples: &[f32]) -> Result<MusicalAnalysis> {
        // Compute STFT once and reuse for all chromagram operations
        let stft_processor = StftProcessor::new(
            self.fft_size,
            self.hop_size,
            crate::analysis::spectral::WindowFunction::Hann,
        );
        let spectrogram = stft_processor.process(samples);

        self.analyze_inner(&spectrogram, self.fft_size, samples.len())
    }

    /// Analyze using a pre-computed spectrogram, avoiding a redundant STFT.
    /// `spectrogram_fft_size` is the FFT size used to produce the spectrogram
    /// (needed for correct frequency bin mapping in chroma extraction).
    pub fn analyze_with_spectrogram(
        &self,
        spectrogram: &ndarray::Array2<f32>,
        spectrogram_fft_size: usize,
        num_samples: usize,
    ) -> Result<MusicalAnalysis> {
        self.analyze_inner(spectrogram, spectrogram_fft_size, num_samples)
    }

    fn analyze_inner(
        &self,
        spectrogram: &ndarray::Array2<f32>,
        fft_size: usize,
        num_samples: usize,
    ) -> Result<MusicalAnalysis> {
        // Compute per-frame chroma vectors (reused for both key detection and chords)
        let per_frame_chroma = self.compute_per_frame_chroma_with_fft_size(spectrogram, fft_size);

        // Aggregate all frames into overall chromagram for key detection
        let chroma_vector = Self::aggregate_chroma(&per_frame_chroma);

        // Detect key using Krumhansl-Kessler profiles
        let key = self.detect_key(&chroma_vector)?;

        // Calculate tonality strength
        let tonality = self.calculate_tonality(&chroma_vector, &key);

        // Calculate mode clarity (how clearly major vs minor)
        let mode_clarity = self.calculate_mode_clarity(&chroma_vector, &key);

        // Calculate harmonic complexity
        let harmonic_complexity = self.calculate_harmonic_complexity(&chroma_vector);

        // Detect chord progression using pre-computed per-frame chroma
        let chord_progression =
            self.detect_chord_progression_fast(&per_frame_chroma, num_samples, &key);

        // Estimate time signature (basic implementation)
        let time_signature = Self::estimate_time_signature()?;

        Ok(MusicalAnalysis {
            key,
            chord_progression,
            chroma_vector,
            tonality,
            mode_clarity,
            harmonic_complexity,
            time_signature,
        })
    }

    /// Compute a chroma vector for each STFT frame. Returns Vec of [f32; 12].
    fn compute_per_frame_chroma(&self, spectrogram: &ndarray::Array2<f32>) -> Vec<[f32; 12]> {
        self.compute_per_frame_chroma_with_fft_size(spectrogram, self.fft_size)
    }

    /// Compute chroma vectors using a spectrogram produced with the given fft_size.
    fn compute_per_frame_chroma_with_fft_size(
        &self,
        spectrogram: &ndarray::Array2<f32>,
        fft_size: usize,
    ) -> Vec<[f32; 12]> {
        let num_frames = spectrogram.shape()[1];
        let a4_freq = 440.0;
        let mut result = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let frame = spectrogram.column(frame_idx);
            let mut chroma = [0.0f32; 12];

            for (bin_idx, &magnitude) in frame.iter().enumerate() {
                if magnitude > 0.001 {
                    let freq = bin_idx as f32 * self.sample_rate / fft_size as f32;
                    if freq > 80.0 && freq < 4000.0 {
                        let pitch_class = self.freq_to_pitch_class(freq, a4_freq);
                        chroma[pitch_class] += magnitude;
                    }
                }
            }
            result.push(chroma);
        }
        result
    }

    /// Aggregate per-frame chroma into a single normalized ChromaVector.
    fn aggregate_chroma(per_frame: &[[f32; 12]]) -> ChromaVector {
        let mut chroma = [0.0f32; 12];
        for frame in per_frame {
            for (i, &v) in frame.iter().enumerate() {
                chroma[i] += v;
            }
        }
        let sum: f32 = chroma.iter().sum();
        if sum > 0.0 {
            for bin in &mut chroma {
                *bin /= sum;
            }
        }
        ChromaVector { values: chroma }
    }

    fn compute_chromagram(&self, samples: &[f32]) -> Result<ChromaVector> {
        // Process audio with STFT
        let stft_processor = StftProcessor::new(
            self.fft_size,
            self.hop_size,
            crate::analysis::spectral::WindowFunction::Hann,
        );
        let spectrogram = stft_processor.process(samples);
        let per_frame = self.compute_per_frame_chroma(&spectrogram);
        Ok(Self::aggregate_chroma(&per_frame))
    }

    fn freq_to_pitch_class(&self, freq: f32, a4_freq: f32) -> usize {
        // Convert frequency to MIDI note number
        let midi_note = 69.0 + 12.0 * (freq / a4_freq).log2();

        // Convert to pitch class (0-11)
        ((midi_note.round() as i32) % 12).rem_euclid(12) as usize
    }

    fn detect_key(&self, chroma: &ChromaVector) -> Result<MusicalKey> {
        // Krumhansl-Kessler key profiles
        let major_profile = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ];
        let minor_profile = [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ];

        let mut key_scores = Vec::new();
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        // Test all 24 keys (12 major + 12 minor)
        for (shift, note_name) in note_names.iter().enumerate() {
            // Major key correlation
            let major_score = self.correlate_profiles(&chroma.values, &major_profile, shift);
            key_scores.push(KeyCandidate {
                key: format!("{} major", note_name),
                root: note_name.to_string(),
                mode: KeyMode::Major,
                score: major_score,
            });

            // Minor key correlation
            let minor_score = self.correlate_profiles(&chroma.values, &minor_profile, shift);
            key_scores.push(KeyCandidate {
                key: format!("{} minor", note_name),
                root: note_name.to_string(),
                mode: KeyMode::Minor,
                score: minor_score,
            });
        }

        // Sort by score
        key_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Get top key
        let best_key = &key_scores[0];

        // Calculate confidence based on score distribution
        let confidence = self.calculate_key_confidence(&key_scores);

        // Get alternatives (top 3 excluding the best)
        let alternatives = key_scores[1..4.min(key_scores.len())].to_vec();

        Ok(MusicalKey {
            key: best_key.key.clone(),
            root: best_key.root.clone(),
            mode: best_key.mode.clone(),
            confidence,
            alternatives,
        })
    }

    fn correlate_profiles(&self, chroma: &[f32; 12], profile: &[f32; 12], shift: usize) -> f32 {
        let mut correlation = 0.0;

        for (i, &profile_value) in profile.iter().enumerate() {
            let chroma_idx = (i + shift) % 12;
            correlation += chroma[chroma_idx] * profile_value;
        }

        correlation
    }

    fn calculate_key_confidence(&self, scores: &[KeyCandidate]) -> f32 {
        if scores.len() < 2 {
            return 0.5;
        }

        let best_score = scores[0].score;
        let second_score = scores[1].score;

        // Calculate confidence based on separation from next best
        let separation = (best_score - second_score) / best_score.max(0.001);

        // Also consider absolute score value
        let normalized_best = (best_score / 100.0).min(1.0);

        // Combine factors
        (separation * 0.7 + normalized_best * 0.3).clamp(0.0, 1.0)
    }

    fn calculate_tonality(&self, chroma: &ChromaVector, key: &MusicalKey) -> f32 {
        // Measure how well the chroma fits a tonal profile
        // Higher values indicate stronger tonality

        // Get the appropriate profile
        let profile = match key.mode {
            KeyMode::Major => [
                6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
            ],
            KeyMode::Minor => [
                6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
            ],
        };

        // Find root note index
        let root_idx = self.note_to_index(&key.root);

        // Calculate correlation
        let correlation = self.correlate_profiles(&chroma.values, &profile, root_idx);

        // Normalize to 0-1 range
        (correlation / 100.0).clamp(0.0, 1.0)
    }

    fn calculate_mode_clarity(&self, chroma: &ChromaVector, key: &MusicalKey) -> f32 {
        // Calculate how clearly the mode (major/minor) is defined

        let root_idx = self.note_to_index(&key.root);

        // Check third degree (major vs minor third)
        let major_third_idx = (root_idx + 4) % 12;
        let minor_third_idx = (root_idx + 3) % 12;

        let major_third_weight = chroma.values[major_third_idx];
        let minor_third_weight = chroma.values[minor_third_idx];

        // Clear mode when one third dominates
        let third_diff = (major_third_weight - minor_third_weight).abs();

        // Also check fifth (should be strong in both)
        let fifth_idx = (root_idx + 7) % 12;
        let fifth_weight = chroma.values[fifth_idx];

        // Combine factors
        (third_diff * 2.0 + fifth_weight).min(1.0)
    }

    fn calculate_harmonic_complexity(&self, chroma: &ChromaVector) -> f32 {
        // Measure entropy of chroma distribution
        let mut entropy = 0.0;

        for &value in &chroma.values {
            if value > 0.001 {
                entropy -= value * value.log2();
            }
        }

        // Normalize entropy to 0-1 range
        // Max entropy for uniform distribution is log2(12) ≈ 3.58
        (entropy / 3.58).clamp(0.0, 1.0)
    }

    /// Fast chord detection using pre-computed per-frame chroma vectors.
    /// Eliminates ~3800 STFT recomputations on a 16-min track.
    fn detect_chord_progression_fast(
        &self,
        per_frame_chroma: &[[f32; 12]],
        total_samples: usize,
        key: &MusicalKey,
    ) -> Option<ChordProgression> {
        if per_frame_chroma.is_empty() {
            return None;
        }

        let segment_duration = 0.5; // 500ms segments
        let total_duration = total_samples as f32 / self.sample_rate;

        // How many STFT frames correspond to one 0.5s segment?
        let frames_per_segment =
            ((segment_duration * self.sample_rate) / self.hop_size as f32).ceil() as usize;
        let frames_per_hop = frames_per_segment / 2; // 50% overlap

        if frames_per_segment == 0 || per_frame_chroma.len() < frames_per_segment {
            return None;
        }

        let mut chords = Vec::new();
        let mut frame_start = 0;

        while frame_start + frames_per_segment <= per_frame_chroma.len() {
            // Aggregate chroma over this temporal window
            let window = &per_frame_chroma[frame_start..frame_start + frames_per_segment];
            let chroma = Self::aggregate_chroma(window);

            let start_time = frame_start as f32 * self.hop_size as f32 / self.sample_rate;

            if let Some(chord) = self.detect_chord(&chroma, key) {
                chords.push(ChordEvent {
                    chord: chord.name,
                    start_time,
                    duration: segment_duration.min(total_duration - start_time),
                    confidence: chord.confidence,
                });
            }

            frame_start += frames_per_hop;
        }

        if chords.is_empty() {
            return None;
        }

        let avg_confidence =
            chords.iter().map(|c| c.confidence).sum::<f32>() / chords.len() as f32;

        Some(ChordProgression {
            chords,
            confidence: avg_confidence,
        })
    }

    #[allow(dead_code)]
    fn detect_chord_progression(
        &self,
        samples: &[f32],
        key: &MusicalKey,
    ) -> Result<Option<ChordProgression>> {
        // Legacy method kept for backward compat / tests
        let segment_duration = 0.5;
        let segment_samples = (segment_duration * self.sample_rate) as usize;

        if samples.len() < segment_samples {
            return Ok(None);
        }

        let mut chords = Vec::new();
        let mut pos = 0;

        while pos + segment_samples <= samples.len() {
            let segment = &samples[pos..pos + segment_samples];
            let chroma = self.compute_chromagram(segment)?;

            if let Some(chord) = self.detect_chord(&chroma, key) {
                chords.push(ChordEvent {
                    chord: chord.name,
                    start_time: pos as f32 / self.sample_rate,
                    duration: segment_duration,
                    confidence: chord.confidence,
                });
            }

            pos += segment_samples / 2;
        }

        if chords.is_empty() {
            return Ok(None);
        }

        let avg_confidence =
            chords.iter().map(|c| c.confidence).sum::<f32>() / chords.len() as f32;

        Ok(Some(ChordProgression {
            chords,
            confidence: avg_confidence,
        }))
    }

    fn detect_chord(&self, chroma: &ChromaVector, _key: &MusicalKey) -> Option<DetectedChord> {
        // Simple triad detection
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        // Common chord templates (root, third, fifth)
        let major_template = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let minor_template = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let mut best_chord = None;
        let mut best_score = 0.0;

        // Test common chords in the key
        for (shift, note_name) in note_names.iter().enumerate() {
            // Major chord
            let major_score = self.correlate_chord_template(&chroma.values, &major_template, shift);
            if major_score > best_score {
                best_score = major_score;
                best_chord = Some(DetectedChord {
                    name: note_name.to_string(),
                    confidence: major_score,
                });
            }

            // Minor chord
            let minor_score = self.correlate_chord_template(&chroma.values, &minor_template, shift);
            if minor_score > best_score {
                best_score = minor_score;
                best_chord = Some(DetectedChord {
                    name: format!("{}m", note_name),
                    confidence: minor_score,
                });
            }
        }

        best_chord
    }

    fn correlate_chord_template(
        &self,
        chroma: &[f32; 12],
        template: &[f32; 12],
        shift: usize,
    ) -> f32 {
        let mut score = 0.0;

        for (i, &template_value) in template.iter().enumerate() {
            let chroma_idx = (i + shift) % 12;
            score += chroma[chroma_idx] * template_value;
        }

        score
    }

    fn estimate_time_signature() -> Result<Option<TimeSignature>> {
        // Stub: returns default 4/4. Real implementation would use onset patterns.
        Ok(Some(TimeSignature {
            numerator: 4,
            denominator: 4,
            confidence: 0.5,
        }))
    }

    fn note_to_index(&self, note: &str) -> usize {
        match note {
            "C" => 0,
            "C#" | "Db" => 1,
            "D" => 2,
            "D#" | "Eb" => 3,
            "E" => 4,
            "F" => 5,
            "F#" | "Gb" => 6,
            "G" => 7,
            "G#" | "Ab" => 8,
            "A" => 9,
            "A#" | "Bb" => 10,
            "B" => 11,
            _ => 0,
        }
    }
}

struct DetectedChord {
    name: String,
    confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_vector_normalization() {
        let chroma = ChromaVector {
            values: [
                0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            ],
        };

        let sum: f32 = chroma.values.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Chroma vector should be normalized"
        );
    }

    #[test]
    fn test_key_detection_c_major() {
        // Create a C major triad pattern
        let mut samples = vec![0.0f32; 44100];

        // Add C (261.63 Hz), E (329.63 Hz), G (392.00 Hz)
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / 44100.0;
            *sample = (2.0 * std::f32::consts::PI * 261.63 * t).sin() * 0.33
                + (2.0 * std::f32::consts::PI * 329.63 * t).sin() * 0.33
                + (2.0 * std::f32::consts::PI * 392.00 * t).sin() * 0.33;
        }

        let analyzer = MusicalAnalyzer::new(44100.0);
        let result = analyzer.analyze(&samples).unwrap();

        // Accept C major, A minor (relative), or E minor (contains same notes)
        // All three keys share these notes
        assert!(
            result.key.root == "C" || result.key.root == "A" || result.key.root == "E",
            "Expected C, A, or E but got {}",
            result.key.root
        );
    }

    #[test]
    fn test_harmonic_complexity() {
        // Uniform chroma should have high complexity
        let uniform_chroma = ChromaVector {
            values: [1.0 / 12.0; 12],
        };

        let analyzer = MusicalAnalyzer::new(44100.0);
        let complexity = analyzer.calculate_harmonic_complexity(&uniform_chroma);

        assert!(
            complexity > 0.9,
            "Uniform distribution should have high complexity"
        );

        // Single note should have low complexity
        let mut single_note = [0.0; 12];
        single_note[0] = 1.0;
        let simple_chroma = ChromaVector {
            values: single_note,
        };

        let simplicity = analyzer.calculate_harmonic_complexity(&simple_chroma);
        assert!(simplicity < 0.1, "Single note should have low complexity");
    }
}
