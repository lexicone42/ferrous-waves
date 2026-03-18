//! Constant-Q chromagram filterbank for pitch class extraction.
//!
//! Instead of mapping linear FFT bins directly to pitch classes (which biases
//! toward low-frequency pitch classes that span more bins), this builds a
//! filterbank of triangular windows centered on each semitone frequency across
//! multiple octaves, then folds them into 12 chroma bins.
//!
//! This is the "pseudo-CQT" approach used by librosa's `chroma_cqt`: start from
//! an STFT spectrogram and apply a log-spaced filterbank to get constant-Q
//! resolution. Each pitch class gets equal representation regardless of octave.

use ndarray::Array2;

/// A filterbank that maps STFT magnitude bins → 12 chroma pitch classes.
///
/// Each pitch class accumulates energy from triangular filters centered on
/// that semitone's frequency across multiple octaves (e.g., A2=110Hz,
/// A3=220Hz, A4=440Hz, A5=880Hz all contribute to pitch class A).
pub struct ChromaFilterBank {
    /// Shape: [12, num_fft_bins]. Each row is the combined filter for one
    /// pitch class, summing triangular windows across all octaves.
    filter_bank: Array2<f32>,
}

impl ChromaFilterBank {
    /// Create a chroma filterbank for the given STFT parameters.
    ///
    /// - `sample_rate`: Audio sample rate in Hz
    /// - `fft_size`: FFT window size (determines frequency bin spacing)
    /// - `min_freq`: Lowest frequency to include (default: ~C2 ≈ 65 Hz)
    /// - `max_freq`: Highest frequency to include (default: ~C7 ≈ 2093 Hz)
    pub fn new(sample_rate: f32, fft_size: usize, min_freq: f32, max_freq: f32) -> Self {
        let filter_bank = Self::build_filter_bank(sample_rate, fft_size, min_freq, max_freq);
        Self { filter_bank }
    }

    /// Create with sensible defaults for music analysis.
    pub fn default_for(sample_rate: f32, fft_size: usize) -> Self {
        // C2 (65.4 Hz) to B6 (1975.5 Hz) — 5 octaves covering most musical content
        // Avoids the very low bass (< C2) where fundamentals overlap with sub-harmonics
        // and the very high treble where harmonics are weak
        Self::new(sample_rate, fft_size, 65.4, 2000.0)
    }

    /// Apply the filterbank to an STFT spectrogram.
    /// Input: [num_fft_bins, num_frames]
    /// Output: [12, num_frames] — per-frame chroma vectors
    pub fn apply(&self, spectrogram: &Array2<f32>) -> Array2<f32> {
        self.filter_bank.dot(spectrogram)
    }

    /// Extract per-frame chroma as Vec<[f32; 12]> for compatibility with existing code.
    pub fn chroma_frames(&self, spectrogram: &Array2<f32>) -> Vec<[f32; 12]> {
        let cq_spec = self.apply(spectrogram);
        let num_frames = cq_spec.shape()[1];
        let mut result = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let col = cq_spec.column(frame_idx);
            let mut chroma = [0.0f32; 12];
            for (i, &val) in col.iter().enumerate() {
                chroma[i] = val;
            }
            result.push(chroma);
        }
        result
    }

    /// Build the filterbank matrix [12, num_fft_bins].
    ///
    /// For each semitone frequency across all octaves in range:
    /// 1. Compute the center FFT bin for that frequency
    /// 2. Create a triangular filter spanning ±1 semitone around the center
    /// 3. Normalize so each semitone contributes equally regardless of octave
    /// 4. Add to the corresponding pitch class row (mod 12)
    fn build_filter_bank(
        sample_rate: f32,
        fft_size: usize,
        min_freq: f32,
        max_freq: f32,
    ) -> Array2<f32> {
        let num_bins = fft_size / 2 + 1;
        let mut bank = Array2::zeros((12, num_bins));
        let a4 = 440.0_f32;

        // Enumerate all semitones from min_freq to max_freq
        // MIDI note for a frequency: 69 + 12 * log2(freq / 440)
        let min_midi = (69.0 + 12.0 * (min_freq / a4).log2()).floor() as i32;
        let max_midi = (69.0 + 12.0 * (max_freq / a4).log2()).ceil() as i32;

        for midi in min_midi..=max_midi {
            let center_freq = a4 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0);
            let pitch_class = midi.rem_euclid(12) as usize;

            // Frequencies one semitone below and above
            let lo_freq = center_freq * 2.0_f32.powf(-1.0 / 12.0);
            let hi_freq = center_freq * 2.0_f32.powf(1.0 / 12.0);

            // Map to FFT bin indices
            let center_bin = (center_freq * fft_size as f32 / sample_rate).round() as usize;
            let lo_bin = (lo_freq * fft_size as f32 / sample_rate).round() as usize;
            let hi_bin = (hi_freq * fft_size as f32 / sample_rate).round() as usize;

            if center_bin >= num_bins || lo_bin >= num_bins {
                continue;
            }
            let hi_bin = hi_bin.min(num_bins - 1);

            // Triangular filter: rises from lo_bin to center_bin, falls to hi_bin
            // Normalize by filter width so each semitone contributes equally
            let width = (hi_bin - lo_bin).max(1) as f32;

            for bin in lo_bin..=center_bin {
                if center_bin > lo_bin {
                    let weight = (bin - lo_bin) as f32 / (center_bin - lo_bin) as f32;
                    bank[[pitch_class, bin]] += weight / width;
                }
            }
            for bin in center_bin..=hi_bin {
                if hi_bin > center_bin {
                    let weight = 1.0 - (bin - center_bin) as f32 / (hi_bin - center_bin) as f32;
                    bank[[pitch_class, bin]] += weight / width;
                }
            }
        }

        // Normalize each pitch class row to unit sum so chroma values are comparable
        for pc in 0..12 {
            let row_sum: f32 = bank.row(pc).sum();
            if row_sum > 1e-10 {
                for bin in 0..num_bins {
                    bank[[pc, bin]] /= row_sum;
                }
            }
        }

        bank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_bank_shape() {
        let bank = ChromaFilterBank::default_for(44100.0, 4096);
        assert_eq!(bank.filter_bank.shape(), &[12, 2049]);
    }

    #[test]
    fn test_filter_bank_rows_normalized() {
        let bank = ChromaFilterBank::default_for(44100.0, 4096);
        for pc in 0..12 {
            let sum: f32 = bank.filter_bank.row(pc).sum();
            // Each row should sum to ~1.0 after normalization
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Pitch class {pc} row sum = {sum}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_filter_bank_no_zero_rows() {
        let bank = ChromaFilterBank::default_for(44100.0, 4096);
        for pc in 0..12 {
            let max: f32 = bank.filter_bank.row(pc).iter().copied().fold(0.0, f32::max);
            assert!(max > 0.0, "Pitch class {pc} has no filter weights");
        }
    }

    #[test]
    fn test_pure_a440_chroma() {
        // Create a spectrogram with energy only at 440 Hz (A4)
        let sr = 44100.0;
        let fft_size = 4096;
        let num_bins = fft_size / 2 + 1;
        let a4_bin = (440.0 * fft_size as f32 / sr).round() as usize;

        let mut spec = Array2::zeros((num_bins, 1));
        spec[[a4_bin, 0]] = 1.0;

        let bank = ChromaFilterBank::default_for(sr, fft_size);
        let chroma = bank.chroma_frames(&spec);

        assert_eq!(chroma.len(), 1);
        // A = pitch class 9 — should have the most energy
        let frame = &chroma[0];
        let max_pc = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_pc, 9, "A440 should map to pitch class 9 (A), got {max_pc}");
    }

    #[test]
    fn test_pure_c_chroma() {
        // Energy at C4 = 261.63 Hz
        let sr = 44100.0;
        let fft_size = 4096;
        let num_bins = fft_size / 2 + 1;
        let c4_bin = (261.63 * fft_size as f32 / sr).round() as usize;

        let mut spec = Array2::zeros((num_bins, 1));
        spec[[c4_bin, 0]] = 1.0;

        let bank = ChromaFilterBank::default_for(sr, fft_size);
        let chroma = bank.chroma_frames(&spec);

        let frame = &chroma[0];
        let max_pc = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_pc, 0, "C261 should map to pitch class 0 (C), got {max_pc}");
    }

    #[test]
    fn test_octave_equivalence() {
        // A at different octaves should all contribute to pitch class 9
        let sr = 44100.0;
        let fft_size = 4096;
        let num_bins = fft_size / 2 + 1;

        let bank = ChromaFilterBank::default_for(sr, fft_size);

        for &freq in &[110.0, 220.0, 440.0, 880.0] {
            let bin = (freq * fft_size as f32 / sr).round() as usize;
            if bin >= num_bins {
                continue;
            }
            let mut spec = Array2::zeros((num_bins, 1));
            spec[[bin, 0]] = 1.0;

            let chroma = bank.chroma_frames(&spec);
            let frame = &chroma[0];
            let max_pc = frame
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                max_pc, 9,
                "A at {freq}Hz should map to pitch class 9, got {max_pc}"
            );
        }
    }
}
