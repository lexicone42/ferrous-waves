use std::path::Path;

use crate::utils::error::{FerrousError, Result};

/// Unified audio decoder using hound (WAV) and puremp3 (MP3).
/// Replaces the previous symphonia-based decoder with focused, pure-Rust crates.
pub struct AudioDecoder {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: usize,
}

impl AudioDecoder {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "wav" | "wave" => Self::decode_wav(path),
            "mp3" => Self::decode_mp3(path),
            _ => Err(FerrousError::AudioDecode(format!(
                "Unsupported format: {}",
                ext
            ))),
        }
    }

    fn decode_wav(path: &Path) -> Result<Self> {
        let reader = hound::WavReader::open(path)
            .map_err(|e| FerrousError::AudioDecode(format!("WAV open error: {}", e)))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as usize;

        let samples = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Int, 16) => reader
                .into_samples::<i16>()
                .map(|s| {
                    s.map(|s| s as f32 / i16::MAX as f32)
                        .map_err(|e| FerrousError::AudioDecode(format!("WAV sample error: {}", e)))
                })
                .collect::<Result<Vec<f32>>>()?,
            (hound::SampleFormat::Int, 24) => reader
                .into_samples::<i32>()
                .map(|s| {
                    s.map(|s| s as f32 / 8_388_608.0) // 2^23
                        .map_err(|e| FerrousError::AudioDecode(format!("WAV sample error: {}", e)))
                })
                .collect::<Result<Vec<f32>>>()?,
            (hound::SampleFormat::Int, 32) => reader
                .into_samples::<i32>()
                .map(|s| {
                    s.map(|s| s as f32 / i32::MAX as f32)
                        .map_err(|e| FerrousError::AudioDecode(format!("WAV sample error: {}", e)))
                })
                .collect::<Result<Vec<f32>>>()?,
            (hound::SampleFormat::Float, _) => reader
                .into_samples::<f32>()
                .map(|s| {
                    s.map_err(|e| FerrousError::AudioDecode(format!("WAV sample error: {}", e)))
                })
                .collect::<Result<Vec<f32>>>()?,
            (fmt, bits) => {
                return Err(FerrousError::AudioDecode(format!(
                    "Unsupported WAV format: {:?} {}bit",
                    fmt, bits
                )))
            }
        };

        Ok(Self {
            samples,
            sample_rate,
            channels,
        })
    }

    fn decode_mp3(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| FerrousError::AudioDecode(format!("MP3 read error: {}", e)))?;

        // puremp3 doesn't handle ID3v2 tags — skip past them to reach MPEG frames
        let audio_data = skip_id3v2(&data);

        let (header, samples_iter) = puremp3::read_mp3(audio_data)
            .map_err(|e| FerrousError::AudioDecode(format!("MP3 decode error: {:?}", e)))?;

        let sample_rate = header.sample_rate.hz();
        let channels = if header.channels == puremp3::Channels::Mono {
            1
        } else {
            2
        };

        // puremp3 yields (f32, f32) stereo pairs
        let mut samples = Vec::new();
        for (left, right) in samples_iter {
            samples.push(left);
            if channels == 2 {
                samples.push(right);
            }
        }

        Ok(Self {
            samples,
            sample_rate,
            channels,
        })
    }

    pub fn decode_all(&mut self) -> Result<Vec<f32>> {
        Ok(std::mem::take(&mut self.samples))
    }

    pub fn sample_rate(&self) -> Option<u32> {
        Some(self.sample_rate)
    }

    pub fn num_channels(&self) -> Option<usize> {
        Some(self.channels)
    }
}

/// Skip past ID3v2 tag header if present. puremp3 expects raw MPEG frames
/// but many MP3 files (especially those with embedded artwork) have large
/// ID3v2 tags that confuse the frame sync search.
fn skip_id3v2(data: &[u8]) -> &[u8] {
    if data.len() >= 10 && &data[0..3] == b"ID3" {
        // ID3v2 tag size is a 4-byte syncsafe integer (7 bits per byte)
        let size = ((data[6] as usize) << 21)
            | ((data[7] as usize) << 14)
            | ((data[8] as usize) << 7)
            | (data[9] as usize);
        let total = size + 10; // 10-byte header + tag body
        if total <= data.len() {
            return &data[total..];
        }
    }
    data
}
