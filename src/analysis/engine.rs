use crate::analysis::classification::{ContentClassification, ContentClassifier};
use crate::analysis::fingerprint::{AudioFingerprint, FingerprintGenerator};
use crate::analysis::musical::{MusicalAnalysis, MusicalAnalyzer};
use crate::analysis::perceptual::{calculate_perceptual_metrics, PerceptualMetrics};
use crate::analysis::pitch::{PitchDetector, PitchTrack, PyinDetector, VibratoAnalysis};
use crate::analysis::quality::{QualityAnalyzer, QualityAssessment};
use crate::analysis::segments::{SegmentAnalysis, SegmentAnalyzer};
use crate::analysis::spectral::{StftProcessor, WindowFunction};
use crate::analysis::temporal::{BeatTracker, OnsetDetector};
use crate::audio::AudioFile;
use crate::cache::{generate_cache_key, Cache};
use crate::utils::error::Result;
use crate::visualization::{RenderData, Renderer};
use serde::{Deserialize, Serialize};
use serde_json;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub summary: AudioSummary,
    pub spectral: SpectralAnalysis,
    pub temporal: TemporalAnalysis,
    pub pitch: PitchAnalysis,
    pub perceptual: PerceptualMetrics,
    pub classification: ContentClassification,
    pub musical: MusicalAnalysis,
    pub quality: QualityAssessment,
    pub segments: SegmentAnalysis,
    pub fingerprint: AudioFingerprint,
    pub visuals: VisualsData,
    pub insights: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSummary {
    pub duration: f32,
    pub sample_rate: u32,
    pub channels: usize,
    pub format: String,
    pub peak_amplitude: f32,
    pub rms_level: f32,
    pub dynamic_range: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysis {
    pub spectral_centroid: Vec<f32>,
    pub spectral_rolloff: Vec<f32>,
    pub spectral_flux: Vec<f32>,
    pub mfcc: Vec<Vec<f32>>,
    pub dominant_frequencies: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysis {
    pub tempo: Option<f32>,
    pub beats: Vec<f32>,
    pub onsets: Vec<f32>,
    pub tempo_stability: f32,
    pub rhythmic_complexity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchAnalysis {
    pub mean_pitch: Option<f32>,
    pub pitch_range: (f32, f32),
    pub pitch_track: PitchTrack,
    pub vibrato: Option<VibratoAnalysis>,
    pub pitch_stability: f32,
    pub dominant_pitch: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualsData {
    pub waveform: Option<String>, // Base64 encoded PNG
    pub spectrogram: Option<String>,
    pub mel_spectrogram: Option<String>,
    pub power_curve: Option<String>,
}

impl AnalysisResult {
    pub fn get_summary(&self) -> serde_json::Value {
        serde_json::json!({
            "duration": self.summary.duration,
            "tempo": self.temporal.tempo,
            "key": self.estimate_key(),
            "energy_profile": self.calculate_energy_profile(),
            "mood_descriptors": self.generate_mood_descriptors(),
        })
    }

    pub fn get_visuals(&self) -> serde_json::Value {
        serde_json::json!({
            "waveform": self.visuals.waveform,
            "spectrogram": self.visuals.spectrogram,
        })
    }

    fn estimate_key(&self) -> String {
        // Use the enhanced musical key detection
        self.musical.key.key.clone()
    }

    fn calculate_energy_profile(&self) -> String {
        match self.summary.rms_level {
            x if x < 0.3 => "low",
            x if x < 0.6 => "medium",
            _ => "high",
        }
        .to_string()
    }

    fn generate_mood_descriptors(&self) -> Vec<String> {
        let mut moods = Vec::new();

        if let Some(tempo) = self.temporal.tempo {
            if tempo < 80.0 {
                moods.push("relaxed".to_string());
            } else if tempo > 140.0 {
                moods.push("energetic".to_string());
            }
        }

        if self.summary.dynamic_range > 20.0 {
            moods.push("dynamic".to_string());
        }

        moods
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct CacheParams {
    fft_size: usize,
    hop_size: usize,
    window_function: WindowFunction,
}

/// Runtime configuration for controlling which analysis steps to perform.
/// All flags default to the full (original) analysis behavior.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Skip PNG visualization generation (waveform, spectrogram, power curve)
    pub skip_visualization: bool,
    /// Skip audio fingerprint generation
    pub skip_fingerprinting: bool,
    /// Skip per-segment content classification (in ContentClassifier)
    pub skip_classification_segments: bool,
    /// Number of PYIN thresholds (default 100, lower = faster but less precise)
    pub pyin_threshold_count: usize,
    /// PYIN hop size multiplier (1 = default 512, 2 = 1024, etc.)
    pub pyin_hop_multiplier: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            skip_visualization: false,
            skip_fingerprinting: false,
            skip_classification_segments: false,
            pyin_threshold_count: 100,
            pyin_hop_multiplier: 1,
        }
    }
}

#[derive(Clone)]
pub struct AnalysisEngine {
    fft_size: usize,
    hop_size: usize,
    window_function: WindowFunction,
    cache: Option<Cache>,
    config: AnalysisConfig,
}

impl AnalysisEngine {
    pub fn new() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            window_function: WindowFunction::Hann,
            cache: Some(Cache::new()),
            config: AnalysisConfig::default(),
        }
    }

    pub fn with_config(fft_size: usize, hop_size: usize, window_function: WindowFunction) -> Self {
        Self {
            fft_size,
            hop_size,
            window_function,
            cache: Some(Cache::new()),
            config: AnalysisConfig::default(),
        }
    }

    pub fn with_cache(mut self, cache: Cache) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn without_cache(mut self) -> Self {
        self.cache = None;
        self
    }

    pub fn with_analysis_config(mut self, config: AnalysisConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn analyze(&self, audio: &AudioFile) -> Result<AnalysisResult> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            let cache_params = CacheParams {
                fft_size: self.fft_size,
                hop_size: self.hop_size,
                window_function: self.window_function,
            };

            let cache_key = generate_cache_key(Path::new(&audio.path), &cache_params);

            // Try to get from cache
            if let Some(cached_data) = cache.get(&cache_key) {
                if let Ok(result) = bincode::deserialize::<AnalysisResult>(&cached_data) {
                    return Ok(result);
                }
            }
        }

        // Convert to mono for analysis
        let mono = audio.buffer.to_mono();

        // Calculate basic metrics
        let peak_amplitude = mono.iter().map(|s| s.abs()).fold(0.0f32, |a, b| a.max(b));

        let rms_level = (mono.iter().map(|s| s * s).sum::<f32>() / mono.len() as f32).sqrt();

        let dynamic_range = if rms_level > 0.0 {
            20.0 * (peak_amplitude / rms_level).log10()
        } else {
            0.0
        };

        // Spectral analysis
        let stft = StftProcessor::new(self.fft_size, self.hop_size, self.window_function);
        let spectrogram = stft.process(&mono);

        // Calculate spectral features
        let mut spectral_centroids = Vec::new();
        let mut spectral_flux = Vec::new();
        let mut dominant_frequencies = Vec::new();

        for frame_idx in 0..spectrogram.shape()[1] {
            let frame = spectrogram.column(frame_idx);

            // Spectral centroid
            let mut weighted_sum = 0.0;
            let mut magnitude_sum = 0.0;
            let mut peak_freq = 0.0;
            let mut peak_mag = 0.0;

            for (bin, &mag) in frame.iter().enumerate() {
                let freq = bin as f32 * audio.buffer.sample_rate as f32 / self.fft_size as f32;
                weighted_sum += freq * mag;
                magnitude_sum += mag;

                if mag > peak_mag {
                    peak_mag = mag;
                    peak_freq = freq;
                }
            }

            if magnitude_sum > 0.0 {
                spectral_centroids.push(weighted_sum / magnitude_sum);
            }

            if frame_idx < 5 {
                dominant_frequencies.push(peak_freq);
            }

            // Spectral flux
            if frame_idx > 0 {
                let prev_frame = spectrogram.column(frame_idx - 1);
                let flux: f32 = frame
                    .iter()
                    .zip(prev_frame.iter())
                    .map(|(&curr, &prev)| (curr - prev).max(0.0).powi(2))
                    .sum();
                spectral_flux.push(flux.sqrt());
            }
        }

        // Temporal analysis
        let onset_detector = OnsetDetector::new();
        let onsets =
            onset_detector.detect_onsets(&spectral_flux, self.hop_size, audio.buffer.sample_rate);

        let beat_tracker = BeatTracker::new();
        let tempo = beat_tracker.estimate_tempo(&onsets);
        let beats = tempo
            .map(|t| beat_tracker.track_beats(&onsets, t))
            .unwrap_or_default();

        // Calculate tempo stability
        let tempo_stability = if beats.len() > 2 {
            let intervals: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).collect();
            let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
            let variance = intervals
                .iter()
                .map(|&i| (i - mean_interval).powi(2))
                .sum::<f32>()
                / intervals.len() as f32;
            1.0 / (1.0 + variance.sqrt())
        } else {
            0.0
        };

        // Generate visualizations (skip if configured)
        let (waveform_base64, spectrogram_base64, power_curve_base64) =
            if self.config.skip_visualization {
                (None, None, None)
            } else {
                let renderer = Renderer::new(1920, 600);
                let wf = renderer.render_to_base64(&RenderData::Waveform(&mono)).ok();
                let sg = renderer
                    .render_to_base64(&RenderData::Spectrogram(&spectrogram))
                    .ok();
                let power_curve: Vec<f32> = (0..spectrogram.shape()[1])
                    .map(|frame_idx| {
                        let frame = spectrogram.column(frame_idx);
                        frame.iter().map(|&x| x * x).sum::<f32>().sqrt()
                    })
                    .collect();
                let pc = renderer
                    .render_to_base64(&RenderData::PowerCurve(&power_curve))
                    .ok();
                (wf, sg, pc)
            };

        // Generate insights
        let mut insights = Vec::new();
        let mut recommendations = Vec::new();

        if peak_amplitude > 0.95 {
            insights.push("Audio contains potential clipping".to_string());
            recommendations.push("Consider reducing input gain to avoid distortion".to_string());
        }

        if let Some(t) = tempo {
            insights.push(format!("Detected tempo: {:.1} BPM", t));
            if t < 80.0 {
                insights
                    .push("Slow tempo detected, suitable for ambient or relaxation".to_string());
            } else if t > 140.0 {
                insights.push("Fast tempo detected, suitable for energetic content".to_string());
            }
        }

        if dynamic_range < 6.0 {
            insights.push("Low dynamic range detected".to_string());
            recommendations
                .push("Consider applying less compression for more dynamic sound".to_string());
        } else if dynamic_range > 20.0 {
            insights.push("High dynamic range preserved".to_string());
        }

        if onsets.len() as f32 / audio.buffer.duration_seconds > 10.0 {
            insights.push("High rhythmic activity detected".to_string());
        }

        // Calculate perceptual metrics
        let perceptual = calculate_perceptual_metrics(&mono, 1, audio.buffer.sample_rate as f32)?;

        // Classify content type (reuse engine's spectrogram, optionally skip per-segment)
        let classifier = ContentClassifier::new(audio.buffer.sample_rate as f32);
        let classification = classifier.classify_with_options(
            &mono,
            Some(&spectrogram),
            self.config.skip_classification_segments,
        )?;

        // Add classification insights
        match classification.primary_type {
            crate::analysis::classification::ContentType::Speech => {
                insights.push(format!(
                    "Content type: Speech (confidence: {:.0}%)",
                    classification.confidence * 100.0
                ));
                if tempo.is_some() {
                    insights.push("Speech detected with rhythmic elements".to_string());
                }
            }
            crate::analysis::classification::ContentType::Music => {
                insights.push(format!(
                    "Content type: Music (confidence: {:.0}%)",
                    classification.confidence * 100.0
                ));
            }
            crate::analysis::classification::ContentType::Silence => {
                insights.push("Content type: Silence detected".to_string());
            }
            crate::analysis::classification::ContentType::Mixed => {
                insights.push(format!(
                    "Content type: Mixed speech/music (speech: {:.0}%, music: {:.0}%)",
                    classification.scores.speech * 100.0,
                    classification.scores.music * 100.0
                ));
            }
        }

        // Perform musical analysis
        let musical_analyzer = MusicalAnalyzer::new(audio.buffer.sample_rate as f32);
        let musical = musical_analyzer.analyze(&mono)?;

        // Assess audio quality
        let quality_analyzer = QualityAnalyzer::new(audio.buffer.sample_rate as f32);
        let quality = quality_analyzer.analyze(&mono)?;

        // Perform segment-based temporal analysis
        let segment_analyzer = SegmentAnalyzer::new(audio.buffer.sample_rate as f32);
        let segments = segment_analyzer.analyze(&mono)?;

        // Generate audio fingerprint (skip if configured)
        let fingerprint = if self.config.skip_fingerprinting {
            AudioFingerprint {
                fingerprint: Vec::new(),
                spectral_hashes: Vec::new(),
                landmarks: Vec::new(),
                perceptual_hash: 0,
                metadata: crate::analysis::fingerprint::FingerprintMetadata {
                    duration: audio.buffer.duration_seconds,
                    sample_rate: audio.buffer.sample_rate as f32,
                    avg_energy: 0.0,
                    dominant_frequencies: Vec::new(),
                    version: 1,
                    params: crate::analysis::fingerprint::FingerprintParams {
                        fft_size: 0,
                        hop_size: 0,
                        num_bands: 0,
                        peak_threshold: 0.0,
                    },
                },
                sub_fingerprints: Vec::new(),
            }
        } else {
            let fingerprint_generator =
                FingerprintGenerator::new(audio.buffer.sample_rate as f32);
            fingerprint_generator.generate(&mono)?
        };

        // Add musical insights
        insights.push(format!(
            "Key: {} (confidence: {:.0}%)",
            musical.key.key,
            musical.key.confidence * 100.0
        ));

        if musical.tonality > 0.7 {
            insights.push("Strong tonal center detected".to_string());
        } else if musical.tonality < 0.3 {
            insights.push("Weak or ambiguous tonality".to_string());
        }

        if musical.harmonic_complexity > 0.7 {
            insights.push("Harmonically complex composition".to_string());
        }

        if let Some(ref progression) = musical.chord_progression {
            if progression.confidence > 0.5 {
                let chord_names: Vec<String> = progression
                    .chords
                    .iter()
                    .take(4)
                    .map(|c| c.chord.clone())
                    .collect();
                if !chord_names.is_empty() {
                    insights.push(format!("Chord progression: {}", chord_names.join(" - ")));
                }
            }
        }

        // Add quality insights
        insights.push(format!(
            "Audio quality score: {:.0}%",
            quality.overall_score * 100.0
        ));

        for issue in quality.issues.iter().take(3) {
            insights.push(format!("Quality issue: {}", issue.description));
        }

        // Add quality recommendations
        recommendations.extend(quality.recommendations.clone());

        // Add segment analysis insights
        insights.push(format!(
            "Temporal structure: {} sections detected",
            segments.structure.len()
        ));

        if segments.temporal_complexity > 0.7 {
            insights.push("Complex temporal structure with varied sections".to_string());
        } else if segments.temporal_complexity < 0.3 {
            insights.push("Simple, repetitive temporal structure".to_string());
        }

        if !segments.patterns.repetitions.is_empty() {
            insights.push(format!(
                "{} repetition patterns found",
                segments.patterns.repetitions.len()
            ));
        }

        if segments.coherence_score > 0.8 {
            insights.push("High segment coherence - smooth transitions".to_string());
        } else if segments.coherence_score < 0.5 {
            insights.push("Low segment coherence - abrupt changes detected".to_string());
        }

        // Add fingerprint insights
        insights.push(format!(
            "Audio fingerprint generated with {} spectral hashes",
            fingerprint.spectral_hashes.len()
        ));

        insights.push(format!(
            "{} acoustic landmarks detected",
            fingerprint.landmarks.len()
        ));

        // Add perceptual insights
        if perceptual.loudness_lufs < -23.0 {
            insights.push(format!(
                "Low loudness level: {:.1} LUFS",
                perceptual.loudness_lufs
            ));
            recommendations
                .push("Consider normalizing to -16 LUFS for streaming platforms".to_string());
        } else if perceptual.loudness_lufs > -14.0 {
            insights.push(format!(
                "High loudness level: {:.1} LUFS",
                perceptual.loudness_lufs
            ));
        }

        if perceptual.true_peak_dbfs > -1.0 {
            insights.push(format!(
                "True peak exceeds recommended level: {:.1} dBFS",
                perceptual.true_peak_dbfs
            ));
            recommendations.push(
                "Consider limiting to -1 dBFS true peak to prevent inter-sample clipping"
                    .to_string(),
            );
        }

        if perceptual.loudness_range > 20.0 {
            insights.push(format!(
                "Wide loudness range: {:.1} LU",
                perceptual.loudness_range
            ));
        } else if perceptual.loudness_range < 5.0 {
            insights.push("Narrow loudness range detected".to_string());
            recommendations.push("Consider adding more dynamic variation".to_string());
        }

        // Pitch detection (configurable threshold count and hop size)
        let pitch_detector =
            PyinDetector::new().with_threshold_count(self.config.pyin_threshold_count);
        let hop_size_pitch = 512 * self.config.pyin_hop_multiplier;
        let pitch_track = pitch_detector.detect_pitch_track(
            &mono,
            audio.buffer.sample_rate as f32,
            hop_size_pitch,
        );

        let valid_pitches: Vec<f32> = pitch_track
            .frames
            .iter()
            .filter_map(|f| f.frequency)
            .collect();

        let (mean_pitch, pitch_range, pitch_stability, dominant_pitch) =
            if !valid_pitches.is_empty() {
                let mean = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;
                let min = valid_pitches.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = valid_pitches
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Calculate pitch stability
                let stability = if valid_pitches.len() > 1 {
                    let diffs: Vec<f32> = valid_pitches
                        .windows(2)
                        .map(|w| (w[1] - w[0]).abs())
                        .collect();
                    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
                    1.0 - (mean_diff / mean).min(1.0)
                } else {
                    0.0
                };

                // Find dominant pitch (most common)
                let mut pitch_counts = std::collections::HashMap::new();
                for &pitch in &valid_pitches {
                    let bin = (pitch / 10.0).round() as i32;
                    *pitch_counts.entry(bin).or_insert(0) += 1;
                }
                let dominant = pitch_counts
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(bin, _)| bin as f32 * 10.0);

                (Some(mean), (min, max), stability, dominant)
            } else {
                (None, (0.0, 0.0), 0.0, None)
            };

        // Vibrato detection
        let vibrato = if !valid_pitches.is_empty() {
            let vibrato_detector = crate::analysis::pitch::VibratoDetector::new();
            vibrato_detector.analyze(
                &valid_pitches,
                audio.buffer.sample_rate as f32,
                hop_size_pitch,
            )
        } else {
            None
        };

        // Add pitch insights
        if let Some(pitch) = mean_pitch {
            insights.push(format!("Average pitch: {:.1} Hz", pitch));
            if pitch_stability > 0.8 {
                insights.push("Stable pitch detected".to_string());
            } else if pitch_stability < 0.5 {
                insights.push("Variable pitch detected".to_string());
            }
        }

        if let Some(ref vib) = vibrato {
            insights.push(format!(
                "Vibrato detected: {:.1} Hz rate, {:.1} cents depth",
                vib.rate, vib.depth_cents
            ));
        }

        let result = AnalysisResult {
            summary: AudioSummary {
                duration: audio.buffer.duration_seconds,
                sample_rate: audio.buffer.sample_rate,
                channels: audio.buffer.channels,
                format: format!("{:?}", audio.format),
                peak_amplitude,
                rms_level,
                dynamic_range,
            },
            spectral: SpectralAnalysis {
                spectral_centroid: spectral_centroids,
                spectral_rolloff: vec![],
                spectral_flux,
                mfcc: vec![],
                dominant_frequencies,
            },
            temporal: TemporalAnalysis {
                tempo,
                beats,
                onsets: onsets.clone(),
                tempo_stability,
                rhythmic_complexity: onsets.len() as f32 / audio.buffer.duration_seconds,
            },
            pitch: PitchAnalysis {
                mean_pitch,
                pitch_range,
                pitch_track,
                vibrato,
                pitch_stability,
                dominant_pitch,
            },
            visuals: VisualsData {
                waveform: waveform_base64,
                spectrogram: spectrogram_base64,
                mel_spectrogram: None,
                power_curve: power_curve_base64,
            },
            perceptual,
            classification,
            musical,
            quality,
            segments,
            fingerprint,
            insights,
            recommendations,
        };

        // Save to cache
        if let Some(ref cache) = self.cache {
            let cache_params = CacheParams {
                fft_size: self.fft_size,
                hop_size: self.hop_size,
                window_function: self.window_function,
            };

            let cache_key = generate_cache_key(Path::new(&audio.path), &cache_params);

            if let Ok(serialized) = bincode::serialize(&result) {
                cache.put(cache_key, serialized).ok();
            }
        }

        Ok(result)
    }

    pub async fn compare(&self, audio_a: &AudioFile, audio_b: &AudioFile) -> ComparisonResult {
        let analysis_a = self.analyze(audio_a).await.ok();
        let analysis_b = self.analyze(audio_b).await.ok();

        let tempo_difference = match (&analysis_a, &analysis_b) {
            (Some(a), Some(b)) => match (a.temporal.tempo, b.temporal.tempo) {
                (Some(ta), Some(tb)) => Some(ta - tb),
                _ => None,
            },
            _ => None,
        };

        // Compare fingerprints
        let (fingerprint_similarity, fingerprint_match_type) = match (&analysis_a, &analysis_b) {
            (Some(a), Some(b)) => {
                let matcher = crate::analysis::fingerprint::FingerprintMatcher::new();
                let match_result = matcher.compare(&a.fingerprint, &b.fingerprint);
                (
                    Some(match_result.similarity),
                    Some(format!("{:?}", match_result.match_type)),
                )
            }
            _ => (None, None),
        };

        let duration_difference = audio_a.buffer.duration_seconds - audio_b.buffer.duration_seconds;
        let sample_rate_match = audio_a.buffer.sample_rate == audio_b.buffer.sample_rate;

        ComparisonResult {
            file_a: FileInfo {
                path: audio_a.path.clone(),
                duration: audio_a.buffer.duration_seconds,
                sample_rate: audio_a.buffer.sample_rate,
                channels: audio_a.buffer.channels,
                tempo: analysis_a.as_ref().and_then(|a| a.temporal.tempo),
            },
            file_b: FileInfo {
                path: audio_b.path.clone(),
                duration: audio_b.buffer.duration_seconds,
                sample_rate: audio_b.buffer.sample_rate,
                channels: audio_b.buffer.channels,
                tempo: analysis_b.as_ref().and_then(|a| a.temporal.tempo),
            },
            comparison: ComparisonMetrics {
                duration_difference,
                sample_rate_match,
                tempo_difference,
                spectral_similarity: None,
                fingerprint_similarity,
                fingerprint_match_type,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub file_a: FileInfo,
    pub file_b: FileInfo,
    pub comparison: ComparisonMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub duration: f32,
    pub sample_rate: u32,
    pub channels: usize,
    pub tempo: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetrics {
    pub duration_difference: f32,
    pub sample_rate_match: bool,
    pub tempo_difference: Option<f32>,
    pub spectral_similarity: Option<f32>,
    pub fingerprint_similarity: Option<f32>,
    pub fingerprint_match_type: Option<String>,
}

impl Default for AnalysisEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export AnalysisConfig for downstream users
impl AnalysisEngine {
    pub fn config(&self) -> &AnalysisConfig {
        &self.config
    }
}
