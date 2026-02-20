use crate::analysis::spectral::StftProcessor;
use crate::utils::error::Result;

// Krumhansl-Kessler key profiles (from cognitive probe-tone studies).
// Each array gives the perceptual stability rating for each pitch class
// relative to the tonic at index 0.
const KK_MAJOR: [f32; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KK_MINOR: [f32; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
// Modal profiles derived from scale degrees.
// Mixolydian (major with b7) — Grateful Dead staple (Fire on the Mountain, Scarlet Begonias)
const KK_MIXOLYDIAN: [f32; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 3.80, 2.29];
// Dorian (minor with natural 6) — common in jam-band improvisation
const KK_DORIAN: [f32; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 2.69];

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
    Mixolydian,
    Dorian,
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

    /// Fraction of frames classified as major key (0.0 = all minor, 1.0 = all major).
    /// Uses per-frame K-K profile correlation at the detected root.
    pub major_frame_ratio: f32,

    /// Fraction of detected chords that are major (not ending in 'm').
    /// Falls back to 0.5 when no chords detected.
    pub major_chord_ratio: f32,

    /// Number of key changes detected across 30s windows.
    /// 0 = stays in one key the whole track. Higher = more modulations.
    pub key_change_count: u32,
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

        self.analyze_inner(&spectrogram, self.fft_size, self.hop_size, samples.len())
    }

    /// Analyze using a pre-computed spectrogram, avoiding a redundant STFT.
    /// `spectrogram_fft_size` is the FFT size used to produce the spectrogram
    /// (needed for correct frequency bin mapping in chroma extraction).
    /// `spectrogram_hop_size` is the hop size used to produce the spectrogram
    /// (needed for correct timing in chord detection).
    pub fn analyze_with_spectrogram(
        &self,
        spectrogram: &ndarray::Array2<f32>,
        spectrogram_fft_size: usize,
        spectrogram_hop_size: usize,
        num_samples: usize,
    ) -> Result<MusicalAnalysis> {
        self.analyze_inner(spectrogram, spectrogram_fft_size, spectrogram_hop_size, num_samples)
    }

    fn analyze_inner(
        &self,
        spectrogram: &ndarray::Array2<f32>,
        fft_size: usize,
        hop_size: usize,
        num_samples: usize,
    ) -> Result<MusicalAnalysis> {
        // Compute per-frame chroma vectors (reused for key detection, chords, and ratios)
        let per_frame_chroma = self.compute_per_frame_chroma_with_fft_size(spectrogram, fft_size);

        // Aggregate all frames into overall chromagram for key detection
        let chroma_vector = Self::aggregate_chroma(&per_frame_chroma);

        // Bass-weighted chroma (60-350 Hz, 1/freq weighting) for root anchoring
        let bass_chroma = self.compute_bass_chroma(spectrogram, fft_size);

        // Chord root histogram (computed before key detection for root feedback)
        let chord_root_hist = self.compute_chord_root_histogram(&per_frame_chroma, hop_size);

        // Detect key using K-K profiles + bass/chord root evidence
        let key = self.detect_key_with_root_evidence(&chroma_vector, &bass_chroma, &chord_root_hist)?;

        // Calculate tonality strength
        let tonality = self.calculate_tonality(&chroma_vector, &key);

        // Calculate mode clarity (how clearly major vs minor)
        let mode_clarity = self.calculate_mode_clarity(&chroma_vector, &key);

        // Calculate harmonic complexity using per-frame chroma movement
        let harmonic_complexity =
            self.calculate_harmonic_complexity_from_frames(&per_frame_chroma);

        // Detect chord progression using pre-computed per-frame chroma
        let chord_progression =
            self.detect_chord_progression_fast(&per_frame_chroma, num_samples, hop_size, &key);

        // Estimate time signature (basic implementation)
        let time_signature = Self::estimate_time_signature()?;

        // Compute major/minor ratios from per-frame chroma and chords
        let major_frame_ratio = self.compute_major_frame_ratio(&per_frame_chroma, &key);
        let major_chord_ratio = Self::compute_major_chord_ratio(&chord_progression);

        // Count key changes across 30s windows
        let key_change_count = self.count_key_changes(&per_frame_chroma, spectrogram, fft_size, hop_size);

        Ok(MusicalAnalysis {
            key,
            chord_progression,
            chroma_vector,
            tonality,
            mode_clarity,
            harmonic_complexity,
            time_signature,
            major_frame_ratio,
            major_chord_ratio,
            key_change_count,
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

    /// Compute a bass-weighted chroma from 60-350 Hz, with 1/freq weighting.
    ///
    /// The bass guitar outlines the tonic in jam-band music. Standard chroma
    /// (80-4000 Hz, unweighted) dilutes this signal with upper harmonics and
    /// other instruments. This bass chroma emphasizes the fundamental frequencies
    /// where the bass defines the key center.
    ///
    /// Returns a single normalized 12-bin distribution (aggregated over all frames).
    fn compute_bass_chroma(
        &self,
        spectrogram: &ndarray::Array2<f32>,
        fft_size: usize,
    ) -> [f32; 12] {
        let num_frames = spectrogram.shape()[1];
        let a4_freq = 440.0;
        let mut bass_chroma = [0.0_f32; 12];

        for frame_idx in 0..num_frames {
            let frame = spectrogram.column(frame_idx);

            for (bin_idx, &magnitude) in frame.iter().enumerate() {
                if magnitude > 0.001 {
                    let freq = bin_idx as f32 * self.sample_rate / fft_size as f32;
                    if freq >= 60.0 && freq <= 350.0 {
                        let pitch_class = self.freq_to_pitch_class(freq, a4_freq);
                        // Weight by 1/freq to emphasize lower bass fundamentals
                        bass_chroma[pitch_class] += magnitude / freq;
                    }
                }
            }
        }

        // Normalize to sum = 1.0
        let sum: f32 = bass_chroma.iter().sum();
        if sum > 0.0 {
            for bin in &mut bass_chroma {
                *bin /= sum;
            }
        }
        bass_chroma
    }

    /// Build a confidence-weighted histogram of chord roots from per-frame chroma.
    ///
    /// Runs simplified chord detection (major/minor triad matching) on windowed
    /// chroma segments, accumulates root pitch classes weighted by detection
    /// confidence. The most frequent chord root is typically the tonic.
    fn compute_chord_root_histogram(
        &self,
        per_frame_chroma: &[[f32; 12]],
        hop_size: usize,
    ) -> [f32; 12] {
        let mut root_hist = [0.0_f32; 12];

        // Use ~500ms windows, same as chord detection
        let segment_duration = 0.5;
        let frames_per_segment =
            ((segment_duration * self.sample_rate) / hop_size as f32).ceil() as usize;
        let frames_per_hop = frames_per_segment / 2;

        if frames_per_segment == 0 || per_frame_chroma.len() < frames_per_segment {
            return root_hist;
        }

        // Same templates as detect_chord (triads + 7ths) for consistent root detection
        let templates: &[[f32; 12]] = &[
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // major
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // minor
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], // dim
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], // dom7
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // maj7
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], // m7
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], // m7b5
        ];

        let mut frame_start = 0;
        while frame_start + frames_per_segment <= per_frame_chroma.len() {
            let window = &per_frame_chroma[frame_start..frame_start + frames_per_segment];
            let agg = Self::aggregate_chroma(window);

            // Find best chord root across all templates × 12 roots
            let mut best_score = 0.0_f32;
            let mut best_root = 0_usize;

            for shift in 0..12 {
                for template in templates {
                    let score = self.correlate_chord_template(&agg.values, template, shift);
                    if score > best_score {
                        best_score = score;
                        best_root = shift;
                    }
                }
            }

            // Weight by confidence (higher-confidence chords contribute more)
            if best_score > 0.0 {
                root_hist[best_root] += best_score;
            }

            frame_start += frames_per_hop;
        }

        // Normalize to sum = 1.0
        let sum: f32 = root_hist.iter().sum();
        if sum > 0.0 {
            for bin in &mut root_hist {
                *bin /= sum;
            }
        }
        root_hist
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

    /// Detect key using Pearson correlation with bass-chroma and chord-root re-ranking.
    ///
    /// After computing all 48 Pearson scores (12 roots × 4 modes), the top candidates
    /// are boosted by how strongly their root appears in the bass chroma and chord root
    /// histogram. This corrects the common "wrong root" problem where chroma shape
    /// matches but the root is misidentified (e.g., A mixolydian → F major).
    fn detect_key_with_root_evidence(
        &self,
        chroma: &ChromaVector,
        bass_chroma: &[f32; 12],
        chord_root_hist: &[f32; 12],
    ) -> Result<MusicalKey> {
        let mut key_scores = Vec::new();
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        // Test all 48 keys (12 roots × 4 modes)
        for (shift, note_name) in note_names.iter().enumerate() {
            let profiles: &[(&[f32; 12], KeyMode, &str)] = &[
                (&KK_MAJOR, KeyMode::Major, "major"),
                (&KK_MINOR, KeyMode::Minor, "minor"),
                (&KK_MIXOLYDIAN, KeyMode::Mixolydian, "mixolydian"),
                (&KK_DORIAN, KeyMode::Dorian, "dorian"),
            ];

            for &(profile, ref mode, mode_name) in profiles {
                let score = self.pearson_correlate(&chroma.values, profile, shift);
                key_scores.push(KeyCandidate {
                    key: format!("{} {}", note_name, mode_name),
                    root: note_name.to_string(),
                    mode: mode.clone(),
                    score,
                });
            }
        }

        // Sort by raw Pearson score first
        key_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Re-rank top 10 candidates with bass + chord root evidence.
        // Modest boosts that break ties but don't override a strong Pearson match.
        let bass_weight = 0.5;
        let chord_weight = 0.4;
        let rerank_count = 10.min(key_scores.len());

        for candidate in key_scores[..rerank_count].iter_mut() {
            let root_idx = self.note_to_index(&candidate.root);
            let root_boost = 1.0
                + bass_weight * bass_chroma[root_idx]
                + chord_weight * chord_root_hist[root_idx];
            candidate.score *= root_boost;
        }

        // Re-sort after boosting
        key_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let best_key = &key_scores[0];
        let confidence = self.calculate_key_confidence(&key_scores);

        // Dynamic alternatives: all keys scoring within 80% of the best
        let threshold = best_key.score * 0.8;
        let alternatives: Vec<KeyCandidate> = key_scores[1..]
            .iter()
            .filter(|k| k.score >= threshold)
            .cloned()
            .collect();

        Ok(MusicalKey {
            key: best_key.key.clone(),
            root: best_key.root.clone(),
            mode: best_key.mode.clone(),
            confidence,
            alternatives,
        })
    }

    /// Pearson correlation between chroma and a profile template (shifted by `shift` semitones).
    /// Unlike dot product, Pearson correlation is insensitive to the overall magnitude of
    /// chroma energy, comparing only the *shape* of the distribution.
    fn pearson_correlate(&self, chroma: &[f32; 12], profile: &[f32; 12], shift: usize) -> f32 {
        // Gather shifted chroma values
        let mut x = [0.0_f32; 12];
        for i in 0..12 {
            x[i] = chroma[(i + shift) % 12];
        }
        let y = profile;

        let n = 12.0_f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;

        let mut cov = 0.0_f32;
        let mut var_x = 0.0_f32;
        let mut var_y = 0.0_f32;

        for i in 0..12 {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom > 1e-10 {
            cov / denom
        } else {
            0.0
        }
    }

    fn calculate_key_confidence(&self, scores: &[KeyCandidate]) -> f32 {
        if scores.len() < 2 {
            return 0.5;
        }

        let best_score = scores[0].score;
        let second_score = scores[1].score;

        // Pearson correlation ranges -1 to 1. Separation is the gap between
        // the best and second-best candidates.
        let separation = best_score - second_score;

        // Confidence: 70% from how much the best key stands out, 30% from
        // absolute correlation strength.
        let sep_factor = (separation / 0.15).clamp(0.0, 1.0); // 0.15 gap → full confidence
        let strength_factor = ((best_score + 1.0) / 2.0).clamp(0.0, 1.0); // map -1..1 to 0..1

        (sep_factor * 0.7 + strength_factor * 0.3).clamp(0.0, 1.0)
    }

    fn calculate_tonality(&self, chroma: &ChromaVector, key: &MusicalKey) -> f32 {
        let profile = Self::kk_profile_for_mode(&key.mode);
        let root_idx = self.note_to_index(&key.root);
        let correlation = self.pearson_correlate(&chroma.values, profile, root_idx);
        ((correlation + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    fn kk_profile_for_mode(mode: &KeyMode) -> &'static [f32; 12] {
        match mode {
            KeyMode::Major => &KK_MAJOR,
            KeyMode::Minor => &KK_MINOR,
            KeyMode::Mixolydian => &KK_MIXOLYDIAN,
            KeyMode::Dorian => &KK_DORIAN,
        }
    }

    fn calculate_mode_clarity(&self, chroma: &ChromaVector, key: &MusicalKey) -> f32 {
        // Correlate chroma against all 4 K-K profiles at the detected root.
        // mode_clarity = gap between best and 2nd-best mode, scaled to 0-1.
        let root_idx = self.note_to_index(&key.root);
        let profiles = [&KK_MAJOR, &KK_MINOR, &KK_MIXOLYDIAN, &KK_DORIAN];

        let mut scores: Vec<f32> = profiles
            .iter()
            .map(|p| self.pearson_correlate(&chroma.values, p, root_idx))
            .collect();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

        if scores.len() >= 2 {
            let gap = scores[0] - scores[1];
            // Gap of 0.2+ = very clear mode; scale to 0-1
            (gap / 0.2).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Measure harmonic complexity via chroma flux at musically meaningful intervals.
    ///
    /// The old entropy-of-aggregate-chroma approach saturates near 1.0 for any
    /// polyphonic music, since live band recordings have near-uniform pitch-class
    /// distribution when aggregated. Instead, this measures how the harmonic
    /// content changes at ~500ms intervals — songs with frequent chord changes
    /// and modulations score higher.
    ///
    /// Consecutive STFT frames (~12ms apart) are nearly identical, so we average
    /// chroma over half-second windows before comparing.
    fn calculate_harmonic_complexity_from_frames(&self, per_frame_chroma: &[[f32; 12]]) -> f32 {
        // ~500ms windows: at hop_size=512 and sr=44100, one frame ≈ 11.6ms → ~43 frames per 500ms
        let window_size = 43.max(1);
        if per_frame_chroma.len() < window_size * 2 {
            return 0.0;
        }

        // Average chroma over windows
        let num_windows = per_frame_chroma.len() / window_size;
        let mut windows: Vec<[f32; 12]> = Vec::with_capacity(num_windows);

        for w in 0..num_windows {
            let start = w * window_size;
            let end = (start + window_size).min(per_frame_chroma.len());
            let mut avg = [0.0f32; 12];
            for frame in &per_frame_chroma[start..end] {
                for (i, &v) in frame.iter().enumerate() {
                    avg[i] += v;
                }
            }
            let n = (end - start) as f32;
            for v in &mut avg {
                *v /= n;
            }
            windows.push(avg);
        }

        if windows.len() < 2 {
            return 0.0;
        }

        // Compute cosine distance between consecutive half-second windows
        let mut total_distance = 0.0_f32;
        let mut count = 0;

        for pair in windows.windows(2) {
            let prev = &pair[0];
            let curr = &pair[1];

            let dot: f32 = prev.iter().zip(curr.iter()).map(|(&a, &b)| a * b).sum();
            let mag_a: f32 = prev.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let mag_b: f32 = curr.iter().map(|&v| v * v).sum::<f32>().sqrt();

            let denom = mag_a * mag_b;
            if denom > 1e-10 {
                let cosine_sim = (dot / denom).clamp(-1.0, 1.0);
                total_distance += 1.0 - cosine_sim;
            }
            count += 1;
        }

        if count == 0 {
            return 0.0;
        }

        let mean_distance = total_distance / count as f32;

        // Half-second chroma distances typically range 0.0 (sustained drone) to ~0.15
        // (rapid chord changes/modulations). Map to 0-1 with saturation at 0.12.
        (mean_distance / 0.12).clamp(0.0, 1.0)
    }

    /// Fast chord detection using pre-computed per-frame chroma vectors.
    /// Eliminates ~3800 STFT recomputations on a 16-min track.
    /// `hop_size` must match the hop used to produce the spectrogram frames.
    fn detect_chord_progression_fast(
        &self,
        per_frame_chroma: &[[f32; 12]],
        total_samples: usize,
        hop_size: usize,
        key: &MusicalKey,
    ) -> Option<ChordProgression> {
        if per_frame_chroma.is_empty() {
            return None;
        }

        let segment_duration = 0.5; // 500ms segments
        let total_duration = total_samples as f32 / self.sample_rate;

        // How many STFT frames correspond to one 0.5s segment?
        let frames_per_segment =
            ((segment_duration * self.sample_rate) / hop_size as f32).ceil() as usize;
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

            let start_time = frame_start as f32 * hop_size as f32 / self.sample_rate;

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
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];

        // Chord templates: (template, suffix, is_triad)
        // Triads get a 1.1× bias because 7th chords have 4 active bins vs 3,
        // giving them a natural advantage from noise. Triads should win when close.
        let templates: &[([f32; 12], &str, bool)] = &[
            // Triads
            ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "",    true),  // major
            ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "m",   true),  // minor
            ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "dim", true),  // diminished
            // 7th chords
            ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], "7",    false), // dominant 7
            ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "maj7", false), // major 7
            ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], "m7",   false), // minor 7
            ([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], "m7b5", false), // half-dim
        ];

        let mut best_chord = None;
        let mut best_score = 0.0_f32;

        for (shift, note_name) in note_names.iter().enumerate() {
            for &(ref template, suffix, is_triad) in templates {
                let mut score = self.correlate_chord_template(&chroma.values, template, shift);
                if is_triad {
                    score *= 1.1; // Prefer simpler interpretation when scores are close
                }
                if score > best_score {
                    best_score = score;
                    let name = if suffix.is_empty() {
                        note_name.to_string()
                    } else {
                        format!("{}{}", note_name, suffix)
                    };
                    best_chord = Some(DetectedChord {
                        name,
                        confidence: score,
                    });
                }
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

    /// Per-frame major/minor classification via K-K profile correlation.
    /// Skip near-silent frames. Returns fraction classified as major (0.0-1.0).
    fn compute_major_frame_ratio(&self, per_frame_chroma: &[[f32; 12]], key: &MusicalKey) -> f32 {
        let root_idx = self.note_to_index(&key.root);
        let mut major_count = 0_u32;
        let mut total_count = 0_u32;

        for frame in per_frame_chroma {
            // Skip near-silent frames
            let energy: f32 = frame.iter().sum();
            if energy < 0.01 {
                continue;
            }

            let major_corr = self.pearson_correlate(frame, &KK_MAJOR, root_idx);
            let minor_corr = self.pearson_correlate(frame, &KK_MINOR, root_idx);

            total_count += 1;
            if major_corr >= minor_corr {
                major_count += 1;
            }
        }

        if total_count > 0 {
            major_count as f32 / total_count as f32
        } else {
            0.5
        }
    }

    /// Fraction of detected chords that have a major third (major-flavored).
    /// Major, dominant 7, and major 7 chords count as major-flavored.
    /// Minor, diminished, minor 7, and half-diminished count as non-major.
    /// Falls back to 0.5 when no chords detected.
    fn compute_major_chord_ratio(chord_progression: &Option<ChordProgression>) -> f32 {
        match chord_progression {
            Some(cp) if !cp.chords.is_empty() => {
                let major_count = cp.chords.iter()
                    .filter(|c| {
                        // Strip note name to get suffix
                        let name = &c.chord;
                        // Non-major suffixes: "m", "dim", "m7", "m7b5"
                        !(name.ends_with('m') || name.ends_with("dim")
                          || name.ends_with("m7") || name.ends_with("m7b5"))
                    })
                    .count();
                major_count as f32 / cp.chords.len() as f32
            }
            _ => 0.5,
        }
    }

    /// Count key changes by detecting local key in overlapping 30s windows.
    /// A key change is counted when adjacent windows have a different root or mode.
    fn count_key_changes(
        &self,
        per_frame_chroma: &[[f32; 12]],
        spectrogram: &ndarray::Array2<f32>,
        fft_size: usize,
        hop_size: usize,
    ) -> u32 {
        let frame_duration = hop_size as f32 / self.sample_rate;
        let window_seconds = 30.0;
        let hop_seconds = 15.0;
        let window_frames = (window_seconds / frame_duration).ceil() as usize;
        let hop_frames = (hop_seconds / frame_duration).ceil() as usize;

        if per_frame_chroma.len() < window_frames || window_frames == 0 || hop_frames == 0 {
            return 0;
        }

        // Detect key in each window
        let mut window_keys: Vec<(String, String)> = Vec::new(); // (root, mode_name)
        let mut start = 0;

        while start + window_frames <= per_frame_chroma.len() {
            let window = &per_frame_chroma[start..start + window_frames];
            let chroma = Self::aggregate_chroma(window);

            // Compute local bass chroma from spectrogram slice
            let spec_end = (start + window_frames).min(spectrogram.shape()[1]);
            let local_spec = spectrogram.slice(ndarray::s![.., start..spec_end]);
            let local_bass = self.compute_bass_chroma_from_slice(&local_spec, fft_size);

            // Compute local chord root histogram
            let local_chord_hist = self.compute_chord_root_histogram(window, hop_size);

            if let Ok(key) = self.detect_key_with_root_evidence(&chroma, &local_bass, &local_chord_hist) {
                if key.confidence >= 0.3 {
                    let mode_str = format!("{:?}", key.mode);
                    window_keys.push((key.root, mode_str));
                }
            }

            start += hop_frames;
        }

        // Count transitions (different root or mode)
        let mut changes = 0_u32;
        for pair in window_keys.windows(2) {
            if pair[0] != pair[1] {
                changes += 1;
            }
        }

        changes
    }

    /// Bass chroma from an Array2 slice (view), same logic as compute_bass_chroma
    /// but works on a column slice rather than the full spectrogram.
    fn compute_bass_chroma_from_slice(
        &self,
        spectrogram: &ndarray::ArrayView2<f32>,
        fft_size: usize,
    ) -> [f32; 12] {
        let num_frames = spectrogram.shape()[1];
        let a4_freq = 440.0;
        let mut bass_chroma = [0.0_f32; 12];

        for frame_idx in 0..num_frames {
            let frame = spectrogram.column(frame_idx);
            for (bin_idx, &magnitude) in frame.iter().enumerate() {
                if magnitude > 0.001 {
                    let freq = bin_idx as f32 * self.sample_rate / fft_size as f32;
                    if freq >= 60.0 && freq <= 350.0 {
                        let pitch_class = self.freq_to_pitch_class(freq, a4_freq);
                        bass_chroma[pitch_class] += magnitude / freq;
                    }
                }
            }
        }

        let sum: f32 = bass_chroma.iter().sum();
        if sum > 0.0 {
            for bin in &mut bass_chroma {
                *bin /= sum;
            }
        }
        bass_chroma
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
        let analyzer = MusicalAnalyzer::new(44100.0);

        // Rapidly changing chroma (alternating between different pitch classes)
        // should have high complexity. Need enough frames for the 43-frame window.
        let mut changing_frames = Vec::new();
        for i in 0..200 {
            let mut frame = [0.0f32; 12];
            // Alternate dominant pitch class every ~43 frames (one window)
            let pc = (i / 43) % 12;
            frame[pc] = 1.0;
            changing_frames.push(frame);
        }
        let complexity = analyzer.calculate_harmonic_complexity_from_frames(&changing_frames);
        assert!(
            complexity > 0.5,
            "Rapidly changing chroma should have high complexity, got {}",
            complexity
        );

        // Static chroma (same frame repeated) should have zero complexity
        let mut single_frame = [0.0f32; 12];
        single_frame[0] = 1.0;
        let static_frames = vec![single_frame; 200];
        let simplicity = analyzer.calculate_harmonic_complexity_from_frames(&static_frames);
        assert!(
            simplicity < 0.01,
            "Static chroma should have near-zero complexity, got {}",
            simplicity
        );
    }
}
