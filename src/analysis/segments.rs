use crate::analysis::classification::ContentType;
#[allow(unused_imports)]
use crate::analysis::spectral::StftProcessor;
use crate::utils::error::Result;
use std::collections::HashMap;

/// Segment-based temporal analysis results
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentAnalysis {
    /// Individual segment analyses
    pub segments: Vec<AudioSegment>,

    /// Structural sections identified
    pub structure: Vec<StructuralSection>,

    /// Temporal patterns detected
    pub patterns: TemporalPatterns,

    /// Transition points between different sections
    pub transitions: Vec<TransitionPoint>,

    /// Overall temporal complexity score
    pub temporal_complexity: f32,

    /// Segment coherence (how similar segments are)
    pub coherence_score: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioSegment {
    /// Start time in seconds
    pub start_time: f32,

    /// Duration in seconds
    pub duration: f32,

    /// Energy level (0.0 to 1.0)
    pub energy: f32,

    /// Spectral centroid (brightness)
    pub spectral_centroid: f32,

    /// Zero crossing rate
    pub zcr: f32,

    /// Dominant content type in this segment
    pub content_type: ContentType,

    /// Musical key if applicable
    pub key: Option<String>,

    /// Tempo if detected
    pub tempo: Option<f32>,

    /// Dynamic range within segment
    pub dynamic_range: f32,

    /// Segment classification label
    pub label: SegmentLabel,

    /// Confidence in segment classification
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SegmentLabel {
    Intro,
    Verse,
    Chorus,
    Bridge,
    Outro,
    Break,
    BuildUp,
    Drop,
    Transition,
    Silence,
    Speech,
    Music,
    Ambient,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StructuralSection {
    /// Section type (e.g., verse, chorus)
    pub section_type: SectionType,

    /// Start time in seconds
    pub start_time: f32,

    /// End time in seconds
    pub end_time: f32,

    /// Segments that make up this section
    pub segment_indices: Vec<usize>,

    /// Average energy in section
    pub avg_energy: f32,

    /// Characteristic features
    pub features: SectionFeatures,

    /// Confidence score
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SectionType {
    Intro,
    Verse,
    Chorus,
    Bridge,
    PreChorus,
    PostChorus,
    Outro,
    Instrumental,
    Solo,
    Break,
    Undefined,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SectionFeatures {
    /// Average spectral characteristics
    pub avg_brightness: f32,
    pub avg_roughness: f32,

    /// Rhythmic density
    pub rhythmic_density: f32,

    /// Harmonic stability
    pub harmonic_stability: f32,

    /// Dynamic variation
    pub dynamic_variation: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TemporalPatterns {
    /// Detected repetition patterns
    pub repetitions: Vec<RepetitionPattern>,

    /// Periodic events (e.g., beats, measures)
    pub periodic_events: Vec<PeriodicEvent>,

    /// Energy envelope shape
    pub energy_profile: EnergyProfile,

    /// Build-up and release patterns
    pub tension_profile: Vec<TensionPoint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RepetitionPattern {
    /// Pattern length in segments
    pub length: usize,

    /// Times the pattern repeats
    pub occurrences: Vec<f32>,

    /// Similarity score between repetitions
    pub similarity: f32,

    /// Pattern type
    pub pattern_type: PatternType,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PatternType {
    Rhythmic,
    Melodic,
    Harmonic,
    Structural,
    Mixed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeriodicEvent {
    /// Period in seconds
    pub period: f32,

    /// Strength of periodicity (0.0 to 1.0)
    pub strength: f32,

    /// Event type
    pub event_type: String,

    /// Phase offset
    pub phase: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnergyProfile {
    /// Overall shape type
    pub shape: EnergyShape,

    /// Energy peaks
    pub peaks: Vec<(f32, f32)>, // (time, energy)

    /// Energy valleys
    pub valleys: Vec<(f32, f32)>,

    /// Average energy
    pub average: f32,

    /// Energy variance
    pub variance: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum EnergyShape {
    Flat,
    Increasing,
    Decreasing,
    Peak,
    Valley,
    Oscillating,
    Complex,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensionPoint {
    /// Time in seconds
    pub time: f32,

    /// Tension level (0.0 to 1.0)
    pub tension: f32,

    /// Type of tension change
    pub change_type: TensionChange,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TensionChange {
    BuildUp,
    Release,
    Plateau,
    Sudden,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransitionPoint {
    /// Time of transition
    pub time: f32,

    /// Type of transition
    pub transition_type: TransitionType,

    /// Strength of transition (0.0 to 1.0)
    pub strength: f32,

    /// Duration of transition
    pub duration: f32,

    /// From segment index
    pub from_segment: usize,

    /// To segment index
    pub to_segment: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TransitionType {
    Smooth,
    Abrupt,
    Crossfade,
    Silence,
    BuildUp,
    DropOut,
    KeyChange,
    TempoChange,
}

/// Analyzer for segment-based temporal analysis
pub struct SegmentAnalyzer {
    sample_rate: f32,
    segment_duration: f32,
    overlap_ratio: f32,
    min_segment_duration: f32,
}

impl SegmentAnalyzer {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            segment_duration: 1.0,      // 1 second default segments
            overlap_ratio: 0.5,         // 50% overlap
            min_segment_duration: 0.25, // 250ms minimum
        }
    }

    pub fn with_segment_duration(mut self, duration: f32) -> Self {
        self.segment_duration = duration.max(self.min_segment_duration);
        self
    }

    pub fn with_overlap(mut self, overlap: f32) -> Self {
        self.overlap_ratio = overlap.clamp(0.0, 0.9);
        self
    }

    pub fn analyze(&self, samples: &[f32]) -> Result<SegmentAnalysis> {
        // Segment the audio
        let segments = self.segment_audio(samples)?;

        // Identify structural sections
        let structure = self.identify_structure(&segments);

        // Detect temporal patterns
        let patterns = self.detect_patterns(&segments, samples);

        // Find transition points
        let transitions = self.find_transitions(&segments);

        // Calculate temporal complexity
        let temporal_complexity = self.calculate_temporal_complexity(&segments, &patterns);

        // Calculate segment coherence
        let coherence_score = self.calculate_coherence(&segments);

        Ok(SegmentAnalysis {
            segments,
            structure,
            patterns,
            transitions,
            temporal_complexity,
            coherence_score,
        })
    }

    fn segment_audio(&self, samples: &[f32]) -> Result<Vec<AudioSegment>> {
        let mut segments = Vec::new();
        let segment_samples = (self.segment_duration * self.sample_rate) as usize;
        let hop_samples = ((1.0 - self.overlap_ratio) * segment_samples as f32) as usize;

        if segment_samples > samples.len() {
            // Single segment for very short audio
            let segment =
                self.analyze_segment(samples, 0.0, samples.len() as f32 / self.sample_rate)?;
            segments.push(segment);
            return Ok(segments);
        }

        let mut pos = 0;
        while pos + segment_samples <= samples.len() {
            let segment_data = &samples[pos..pos + segment_samples];
            let start_time = pos as f32 / self.sample_rate;
            let duration = segment_samples as f32 / self.sample_rate;

            let segment = self.analyze_segment(segment_data, start_time, duration)?;
            segments.push(segment);

            pos += hop_samples;
        }

        // Handle last segment if there's remaining data
        if pos < samples.len()
            && samples.len() - pos > (self.min_segment_duration * self.sample_rate) as usize
        {
            let segment_data = &samples[pos..];
            let start_time = pos as f32 / self.sample_rate;
            let duration = (samples.len() - pos) as f32 / self.sample_rate;

            let segment = self.analyze_segment(segment_data, start_time, duration)?;
            segments.push(segment);
        }

        Ok(segments)
    }

    fn analyze_segment(
        &self,
        samples: &[f32],
        start_time: f32,
        duration: f32,
    ) -> Result<AudioSegment> {
        // Calculate energy (cheap — time domain only)
        let energy = self.calculate_energy(samples);

        // Calculate spectral centroid using lightweight method (no STFT)
        let spectral_centroid = self.calculate_spectral_centroid_fast(samples);

        // Calculate zero crossing rate (cheap — time domain only)
        let zcr = self.calculate_zcr(samples);

        // Lightweight content type estimation from time-domain features only.
        // Avoids creating ContentClassifier + MusicalAnalyzer per segment
        // which would each compute their own STFTs.
        let content_type = if energy < 0.01 {
            ContentType::Silence
        } else {
            ContentType::Music // conservative default for jam-band audio
        };

        // Skip per-segment key detection (extremely expensive — creates MusicalAnalyzer
        // with full STFT + chord detection per segment). Key is detected once globally.
        let key = None;

        // Detect tempo (simplified - uses zero crossings)
        let tempo = self.detect_segment_tempo(samples);

        // Calculate dynamic range
        let dynamic_range = self.calculate_dynamic_range(samples);

        // Classify segment label
        let (label, confidence) = self.classify_segment(
            energy,
            spectral_centroid,
            zcr,
            &content_type,
            dynamic_range,
        );

        Ok(AudioSegment {
            start_time,
            duration,
            energy,
            spectral_centroid,
            zcr,
            content_type,
            key,
            tempo,
            dynamic_range,
            label,
            confidence,
        })
    }

    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum: f32 = samples.iter().map(|x| x * x).sum();
        (sum / samples.len() as f32).sqrt()
    }

    /// Fast spectral centroid approximation using zero crossing rate.
    /// ZCR correlates with spectral centroid for most audio signals
    /// and avoids a full STFT computation per segment.
    fn calculate_spectral_centroid_fast(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        // Estimate centroid from ZCR: centroid ≈ zcr * sample_rate / 2
        let mut crossings = 0u32;
        for i in 1..samples.len() {
            if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
                crossings += 1;
            }
        }

        let zcr = crossings as f32 / samples.len() as f32;
        zcr * self.sample_rate / 2.0
    }

    #[allow(dead_code)]
    fn calculate_spectral_centroid(&self, samples: &[f32]) -> f32 {
        let stft = StftProcessor::new(2048, 512, crate::analysis::spectral::WindowFunction::Hann);
        let spectrogram = stft.process(samples);

        if spectrogram.is_empty() {
            return 0.0;
        }

        let mut centroid_sum = 0.0;
        let mut magnitude_sum = 0.0;

        for frame in spectrogram.columns() {
            for (bin, &mag) in frame.iter().enumerate() {
                let freq = bin as f32 * self.sample_rate / 2048.0;
                centroid_sum += freq * mag;
                magnitude_sum += mag;
            }
        }

        if magnitude_sum > 0.0 {
            centroid_sum / magnitude_sum
        } else {
            0.0
        }
    }

    fn calculate_zcr(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (samples.len() - 1) as f32
    }

    fn calculate_dynamic_range(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Calculate RMS over small windows
        let window_size = (self.sample_rate * 0.05) as usize; // 50ms windows
        let mut rms_values = Vec::new();

        for chunk in samples.chunks(window_size) {
            let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            if rms > 0.0 {
                rms_values.push(rms);
            }
        }

        if rms_values.len() < 2 {
            return 0.0;
        }

        rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let high_idx = (rms_values.len() as f32 * 0.95) as usize;
        let low_idx = (rms_values.len() as f32 * 0.05) as usize;

        let high_level = rms_values[high_idx.min(rms_values.len() - 1)];
        let low_level = rms_values[low_idx];

        if low_level > 0.0 {
            20.0 * (high_level / low_level).log10()
        } else {
            40.0
        }
    }

    fn detect_segment_tempo(&self, _samples: &[f32]) -> Option<f32> {
        // Simplified - in full implementation would use beat tracker
        None
    }

    fn classify_segment(
        &self,
        energy: f32,
        spectral_centroid: f32,
        _zcr: f32,
        content_type: &ContentType,
        dynamic_range: f32,
    ) -> (SegmentLabel, f32) {
        match content_type {
            ContentType::Silence => (SegmentLabel::Silence, 0.95),
            ContentType::Speech => (SegmentLabel::Speech, 0.8),
            ContentType::Music => {
                // Classify musical segments based on features
                if energy < 0.1 {
                    (SegmentLabel::Break, 0.7)
                } else if energy > 0.7 && spectral_centroid > 2000.0 {
                    (SegmentLabel::Drop, 0.6)
                } else if dynamic_range > 20.0 && energy > 0.5 {
                    (SegmentLabel::Chorus, 0.5)
                } else if spectral_centroid < 1000.0 && energy < 0.3 {
                    (SegmentLabel::Ambient, 0.6)
                } else {
                    (SegmentLabel::Verse, 0.4)
                }
            }
            ContentType::Mixed => {
                if energy > 0.6 {
                    (SegmentLabel::Music, 0.5)
                } else {
                    (SegmentLabel::Transition, 0.6)
                }
            }
        }
    }

    fn identify_structure(&self, segments: &[AudioSegment]) -> Vec<StructuralSection> {
        let mut sections = Vec::new();
        if segments.is_empty() {
            return sections;
        }

        let mut current_section = vec![0];
        let mut current_label = &segments[0].label;

        for (i, segment) in segments.iter().enumerate().skip(1) {
            if &segment.label != current_label
                || self.is_significant_change(segment, &segments[i - 1])
            {
                // End current section and start new one
                if current_section.len() >= 2 {
                    let section = self.create_section(segments, current_section.clone());
                    sections.push(section);
                }
                current_section = vec![i];
                current_label = &segment.label;
            } else {
                current_section.push(i);
            }
        }

        // Add final section
        if current_section.len() >= 2 {
            let section = self.create_section(segments, current_section);
            sections.push(section);
        }

        sections
    }

    fn is_significant_change(&self, current: &AudioSegment, previous: &AudioSegment) -> bool {
        // Check for significant changes in features
        let energy_change = (current.energy - previous.energy).abs() > 0.3;
        let spectral_change =
            (current.spectral_centroid - previous.spectral_centroid).abs() > 1000.0;
        let content_change = current.content_type != previous.content_type;

        energy_change || spectral_change || content_change
    }

    fn create_section(&self, segments: &[AudioSegment], indices: Vec<usize>) -> StructuralSection {
        let start_time = segments[indices[0]].start_time;
        let end_time = segments[indices[indices.len() - 1]].start_time
            + segments[indices[indices.len() - 1]].duration;

        // Calculate average features
        let mut avg_energy = 0.0;
        let mut avg_brightness = 0.0;
        let mut avg_dynamic = 0.0;

        for &idx in &indices {
            avg_energy += segments[idx].energy;
            avg_brightness += segments[idx].spectral_centroid;
            avg_dynamic += segments[idx].dynamic_range;
        }

        let count = indices.len() as f32;
        avg_energy /= count;
        avg_brightness /= count;
        avg_dynamic /= count;

        // Determine section type based on dominant label
        let section_type = self.determine_section_type(segments, &indices);

        let features = SectionFeatures {
            avg_brightness,
            avg_roughness: 0.5,      // Placeholder
            rhythmic_density: 0.5,   // Placeholder
            harmonic_stability: 0.7, // Placeholder
            dynamic_variation: avg_dynamic,
        };

        StructuralSection {
            section_type,
            start_time,
            end_time,
            segment_indices: indices,
            avg_energy,
            features,
            confidence: 0.6, // Placeholder confidence
        }
    }

    fn determine_section_type(&self, segments: &[AudioSegment], indices: &[usize]) -> SectionType {
        // Count label occurrences
        let mut label_counts = HashMap::new();
        for &idx in indices {
            *label_counts.entry(&segments[idx].label).or_insert(0) += 1;
        }

        // Find dominant label
        let dominant_label = label_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| *label)
            .unwrap_or(&SegmentLabel::Music);

        // Map segment label to section type
        match dominant_label {
            SegmentLabel::Intro => SectionType::Intro,
            SegmentLabel::Verse => SectionType::Verse,
            SegmentLabel::Chorus => SectionType::Chorus,
            SegmentLabel::Bridge => SectionType::Bridge,
            SegmentLabel::Outro => SectionType::Outro,
            SegmentLabel::Break => SectionType::Break,
            _ => SectionType::Undefined,
        }
    }

    fn detect_patterns(&self, segments: &[AudioSegment], _samples: &[f32]) -> TemporalPatterns {
        let repetitions = self.find_repetitions(segments);
        let periodic_events = self.find_periodic_events(segments);
        let energy_profile = self.analyze_energy_profile(segments);
        let tension_profile = self.analyze_tension(segments);

        TemporalPatterns {
            repetitions,
            periodic_events,
            energy_profile,
            tension_profile,
        }
    }

    fn find_repetitions(&self, segments: &[AudioSegment]) -> Vec<RepetitionPattern> {
        let mut patterns = Vec::new();

        // Simple repetition detection based on segment similarity
        for window_size in 2..=8 {
            if window_size > segments.len() / 2 {
                break;
            }

            for i in 0..segments.len() - window_size {
                let pattern_a = &segments[i..i + window_size];

                for j in i + window_size..segments.len() - window_size {
                    let pattern_b = &segments[j..j + window_size];

                    let similarity = self.calculate_pattern_similarity(pattern_a, pattern_b);

                    if similarity > 0.7 {
                        patterns.push(RepetitionPattern {
                            length: window_size,
                            occurrences: vec![segments[i].start_time, segments[j].start_time],
                            similarity,
                            pattern_type: PatternType::Structural,
                        });
                    }
                }
            }
        }

        patterns
    }

    fn calculate_pattern_similarity(
        &self,
        pattern_a: &[AudioSegment],
        pattern_b: &[AudioSegment],
    ) -> f32 {
        if pattern_a.len() != pattern_b.len() {
            return 0.0;
        }

        let mut similarity = 0.0;
        for (a, b) in pattern_a.iter().zip(pattern_b) {
            let energy_sim = 1.0 - (a.energy - b.energy).abs();
            let spectral_sim =
                1.0 - ((a.spectral_centroid - b.spectral_centroid).abs() / 10000.0).min(1.0);
            let label_sim = if a.label == b.label { 1.0 } else { 0.0 };

            similarity += (energy_sim + spectral_sim + label_sim) / 3.0;
        }

        similarity / pattern_a.len() as f32
    }

    fn find_periodic_events(&self, segments: &[AudioSegment]) -> Vec<PeriodicEvent> {
        let mut events = Vec::new();

        // Find energy peaks
        let energy_peaks: Vec<f32> = segments
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                *i > 0
                    && *i < segments.len() - 1
                    && s.energy > segments[i - 1].energy
                    && s.energy > segments[i + 1].energy
            })
            .map(|(_, s)| s.start_time)
            .collect();

        if energy_peaks.len() > 2 {
            // Calculate intervals between peaks
            let mut intervals = Vec::new();
            for i in 1..energy_peaks.len() {
                intervals.push(energy_peaks[i] - energy_peaks[i - 1]);
            }

            // Find dominant period
            if !intervals.is_empty() {
                intervals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_period = intervals[intervals.len() / 2];

                // Compute strength as fraction of intervals within ±10% of median
                let tolerance = median_period * 0.1;
                let consistent_count = intervals
                    .iter()
                    .filter(|&&iv| (iv - median_period).abs() <= tolerance)
                    .count();
                let strength = consistent_count as f32 / intervals.len() as f32;

                events.push(PeriodicEvent {
                    period: median_period,
                    strength,
                    event_type: "Energy Peak".to_string(),
                    phase: energy_peaks[0] % median_period,
                });
            }
        }

        events
    }

    fn analyze_energy_profile(&self, segments: &[AudioSegment]) -> EnergyProfile {
        let energies: Vec<f32> = segments.iter().map(|s| s.energy).collect();

        if energies.is_empty() {
            return EnergyProfile {
                shape: EnergyShape::Flat,
                peaks: vec![],
                valleys: vec![],
                average: 0.0,
                variance: 0.0,
            };
        }

        let average = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance =
            energies.iter().map(|e| (e - average).powi(2)).sum::<f32>() / energies.len() as f32;

        // Find peaks and valleys
        let mut peaks = Vec::new();
        let mut valleys = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            if i > 0 && i < segments.len() - 1 {
                if segment.energy > segments[i - 1].energy
                    && segment.energy > segments[i + 1].energy
                {
                    peaks.push((segment.start_time, segment.energy));
                } else if segment.energy < segments[i - 1].energy
                    && segment.energy < segments[i + 1].energy
                {
                    valleys.push((segment.start_time, segment.energy));
                }
            }
        }

        // Determine overall shape
        let shape = self.determine_energy_shape(&energies);

        EnergyProfile {
            shape,
            peaks,
            valleys,
            average,
            variance,
        }
    }

    fn determine_energy_shape(&self, energies: &[f32]) -> EnergyShape {
        if energies.len() < 3 {
            return EnergyShape::Flat;
        }

        let first_third = energies.len() / 3;
        let last_third = energies.len() * 2 / 3;

        let avg_first = energies[..first_third].iter().sum::<f32>() / first_third as f32;
        let avg_last =
            energies[last_third..].iter().sum::<f32>() / (energies.len() - last_third) as f32;

        let diff = avg_last - avg_first;

        if diff.abs() < 0.1 {
            EnergyShape::Flat
        } else if diff > 0.2 {
            EnergyShape::Increasing
        } else if diff < -0.2 {
            EnergyShape::Decreasing
        } else {
            EnergyShape::Complex
        }
    }

    fn analyze_tension(&self, segments: &[AudioSegment]) -> Vec<TensionPoint> {
        let mut tension_points = Vec::new();

        for (i, segment) in segments.iter().enumerate() {
            if i == 0 {
                continue;
            }

            let prev_segment = &segments[i - 1];

            // Calculate tension based on various factors
            let energy_change = segment.energy - prev_segment.energy;
            let spectral_change =
                (segment.spectral_centroid - prev_segment.spectral_centroid) / 1000.0;

            let tension = (energy_change.abs() + spectral_change.abs() / 10.0).clamp(0.0, 1.0);

            let change_type = if energy_change > 0.2 {
                TensionChange::BuildUp
            } else if energy_change < -0.2 {
                TensionChange::Release
            } else {
                TensionChange::Plateau
            };

            tension_points.push(TensionPoint {
                time: segment.start_time,
                tension,
                change_type,
            });
        }

        tension_points
    }

    fn find_transitions(&self, segments: &[AudioSegment]) -> Vec<TransitionPoint> {
        let mut transitions = Vec::new();

        for i in 1..segments.len() {
            let prev = &segments[i - 1];
            let curr = &segments[i];

            // Check for significant changes
            let energy_diff = (curr.energy - prev.energy).abs();
            let spectral_diff = (curr.spectral_centroid - prev.spectral_centroid).abs() / 5000.0;
            let content_change = prev.content_type != curr.content_type;

            if energy_diff > 0.3 || spectral_diff > 0.3 || content_change {
                let transition_type = if content_change {
                    TransitionType::Abrupt
                } else if energy_diff > 0.5 {
                    if curr.energy > prev.energy {
                        TransitionType::BuildUp
                    } else {
                        TransitionType::DropOut
                    }
                } else {
                    TransitionType::Smooth
                };

                let strength = (energy_diff + spectral_diff).min(1.0);

                transitions.push(TransitionPoint {
                    time: curr.start_time,
                    transition_type,
                    strength,
                    duration: 0.1, // Simplified
                    from_segment: i - 1,
                    to_segment: i,
                });
            }
        }

        transitions
    }

    fn calculate_temporal_complexity(
        &self,
        segments: &[AudioSegment],
        patterns: &TemporalPatterns,
    ) -> f32 {
        if segments.is_empty() {
            return 0.0;
        }

        let energies: Vec<f32> = segments.iter().map(|s| s.energy).collect();
        let mean_energy = energies.iter().sum::<f32>() / energies.len() as f32;

        // Factor 1 (0.2): Energy range — how wide the dynamic spread is
        let min_e = energies.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_e = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let energy_range = if max_e > 1e-10 {
            ((max_e - min_e) / max_e).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Factor 2 (0.2): Energy CV — coefficient of variation
        let energy_cv = if mean_energy > 1e-10 {
            let var = energies.iter().map(|&e| (e - mean_energy).powi(2)).sum::<f32>()
                / energies.len() as f32;
            (var.sqrt() / mean_energy).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Factor 3 (0.4): Transition rate per segment (transitions = tension profile entries)
        let n_seg = segments.len() as f32;
        let transition_rate = (patterns.tension_profile.len() as f32 / n_seg).clamp(0.0, 1.0);

        // Factor 4 (0.2): Repetition density — more repetitions = less complex
        let repetition_density = 1.0 - (patterns.repetitions.len() as f32 / n_seg).clamp(0.0, 1.0);

        let complexity = energy_range * 0.2
            + energy_cv * 0.2
            + transition_rate * 0.4
            + repetition_density * 0.2;

        complexity.clamp(0.0, 1.0)
    }

    fn calculate_coherence(&self, segments: &[AudioSegment]) -> f32 {
        if segments.len() < 2 {
            return 1.0;
        }

        let mut coherence_sum = 0.0;
        let mut count = 0;

        for i in 1..segments.len() {
            let similarity = self.calculate_segment_similarity(&segments[i - 1], &segments[i]);
            coherence_sum += similarity;
            count += 1;
        }

        coherence_sum / count as f32
    }

    fn calculate_segment_similarity(&self, seg_a: &AudioSegment, seg_b: &AudioSegment) -> f32 {
        let energy_sim = 1.0 - (seg_a.energy - seg_b.energy).abs();
        let spectral_sim = 1.0
            - ((seg_a.spectral_centroid - seg_b.spectral_centroid) / 10000.0)
                .abs()
                .min(1.0);
        let zcr_sim = 1.0 - (seg_a.zcr - seg_b.zcr).abs().min(1.0);

        (energy_sim + spectral_sim + zcr_sim) / 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_analyzer_creation() {
        let analyzer = SegmentAnalyzer::new(44100.0);
        assert_eq!(analyzer.sample_rate, 44100.0);
        assert_eq!(analyzer.segment_duration, 1.0);
        assert_eq!(analyzer.overlap_ratio, 0.5);
    }

    #[test]
    fn test_segment_analysis_short_audio() {
        let samples = vec![0.0; 44100]; // 1 second of silence
        let analyzer = SegmentAnalyzer::new(44100.0);
        let result = analyzer.analyze(&samples).unwrap();

        assert!(!result.segments.is_empty());
        assert_eq!(result.segments[0].content_type, ContentType::Silence);
    }

    #[test]
    fn test_energy_calculation() {
        let analyzer = SegmentAnalyzer::new(44100.0);

        // Test with silence
        let silence = vec![0.0; 1000];
        assert_eq!(analyzer.calculate_energy(&silence), 0.0);

        // Test with constant signal
        let constant = vec![0.5; 1000];
        let energy = analyzer.calculate_energy(&constant);
        assert!((energy - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_zcr_calculation() {
        let analyzer = SegmentAnalyzer::new(44100.0);

        // Test with no crossings
        let no_cross = vec![1.0; 1000];
        assert_eq!(analyzer.calculate_zcr(&no_cross), 0.0);

        // Test with alternating signs
        let alternating: Vec<f32> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let zcr = analyzer.calculate_zcr(&alternating);
        assert!(zcr > 0.9); // Should be close to 1.0
    }

    #[test]
    fn test_temporal_patterns() {
        let samples = vec![0.1; 88200]; // 2 seconds
        let analyzer = SegmentAnalyzer::new(44100.0);
        let result = analyzer.analyze(&samples).unwrap();

        assert_eq!(result.patterns.energy_profile.shape, EnergyShape::Flat);
        assert!(result.patterns.energy_profile.variance < 0.01);
    }
}
