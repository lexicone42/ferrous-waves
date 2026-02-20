use ferrous_waves::analysis::temporal::{BeatTracker, OnsetDetector};

#[test]
fn test_onset_detector_creation() {
    OnsetDetector::new();
    // Should create without panic
}

#[test]
fn test_onset_detection_empty_signal() {
    let detector = OnsetDetector::new();
    let onsets = detector.detect_onsets(&[], 512, 44100);
    assert_eq!(onsets.len(), 0);
}

#[test]
fn test_onset_detection_impulses() {
    let detector = OnsetDetector::new();

    // Create spectral flux with clear peaks
    let mut flux = vec![0.1; 100];
    flux[10] = 1.0; // Strong onset at frame 10
    flux[30] = 1.2; // Strong onset at frame 30
    flux[60] = 0.9; // Strong onset at frame 60

    let onsets = detector.detect_onsets(&flux, 512, 44100);

    // Should detect at least some of the peaks
    assert!(onsets.len() >= 2, "Should detect at least 2 onsets");

    // Convert back to frame indices for checking
    let frame_indices: Vec<usize> = onsets
        .iter()
        .map(|&time| (time * 44100.0 / 512.0) as usize)
        .collect();

    // Check if major peaks were detected (with some tolerance)
    let has_first = frame_indices.iter().any(|&i| (i as i32 - 10).abs() < 3);
    let has_second = frame_indices.iter().any(|&i| (i as i32 - 30).abs() < 3);

    assert!(
        has_first || has_second,
        "Should detect at least one major peak"
    );
}

#[test]
fn test_onset_detection_regular_beats() {
    let detector = OnsetDetector::new();

    // Create regular beat pattern
    let mut flux = vec![0.1; 200];
    for i in (0..200).step_by(20) {
        flux[i] = 1.0;
    }

    let onsets = detector.detect_onsets(&flux, 512, 44100);

    // Should detect regular pattern
    assert!(onsets.len() >= 5, "Should detect multiple regular beats");

    // Check regularity of detected onsets
    if onsets.len() >= 2 {
        let intervals: Vec<f32> = onsets.windows(2).map(|w| w[1] - w[0]).collect();

        // Intervals should be relatively consistent
        let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        for interval in &intervals {
            let deviation = (interval - mean_interval).abs() / mean_interval;
            assert!(deviation < 0.3, "Intervals should be regular");
        }
    }
}

#[test]
fn test_beat_tracker_creation() {
    BeatTracker::new();
    // Should create without panic
}

#[test]
fn test_tempo_estimation_empty() {
    let tracker = BeatTracker::new();
    let tempo = tracker.estimate_tempo(&[]);
    assert!(tempo.is_none());

    let tempo = tracker.estimate_tempo(&[1.0]);
    assert!(tempo.is_none());
}

#[test]
fn test_tempo_estimation_regular_beats() {
    let tracker = BeatTracker::new();

    // Create onsets at 120 BPM (0.5 seconds apart)
    let onsets = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let tempo = tracker.estimate_tempo(&onsets);

    assert!(tempo.is_some());
    let bpm = tempo.unwrap();
    assert!(
        (bpm - 120.0).abs() < 5.0,
        "Should detect ~120 BPM, got {}",
        bpm
    );
}

#[test]
fn test_tempo_estimation_different_tempos() {
    let tracker = BeatTracker::new();

    // Test 60 BPM (1 second apart)
    let onsets_60 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let tempo_60 = tracker.estimate_tempo(&onsets_60).unwrap();
    assert!((tempo_60 - 60.0).abs() < 5.0, "Should detect ~60 BPM");

    // Test 180 BPM (0.333 seconds apart)
    let onsets_180: Vec<f32> = (0..10).map(|i| i as f32 * 0.333).collect();
    let tempo_180 = tracker.estimate_tempo(&onsets_180).unwrap();
    assert!((tempo_180 - 180.0).abs() < 10.0, "Should detect ~180 BPM");
}

#[test]
fn test_beat_tracking_empty() {
    let tracker = BeatTracker::new();
    let beats = tracker.track_beats(&[], 120.0);
    assert_eq!(beats.len(), 0);
}

#[test]
fn test_beat_tracking_alignment() {
    let tracker = BeatTracker::new();

    // Create onsets with slight variations around 120 BPM
    let onsets = vec![0.0, 0.48, 1.02, 1.49, 2.01, 2.52, 2.98, 3.51, 4.0];

    let beats = tracker.track_beats(&onsets, 120.0);

    assert!(!beats.is_empty(), "Should generate beats");

    // Beat period for 120 BPM is 0.5 seconds
    if beats.len() >= 2 {
        let intervals: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).collect();

        for interval in intervals {
            assert!(
                (interval - 0.5).abs() < 0.01,
                "Beat intervals should be ~0.5s"
            );
        }
    }
}

#[test]
fn test_beat_tracking_phase_detection() {
    let tracker = BeatTracker::new();

    // Onsets that start at 0.25s offset (not at 0)
    let onsets = vec![0.25, 0.75, 1.25, 1.75, 2.25, 2.75];
    let beats = tracker.track_beats(&onsets, 120.0);

    assert!(!beats.is_empty());

    // First beat should be close to 0.25 (the phase offset)
    if !beats.is_empty() {
        assert!((beats[0] - 0.25).abs() < 0.1, "Should detect phase offset");
    }
}

#[test]
fn test_tempo_range_limits() {
    let tracker = BeatTracker::new();

    // Very fast tempo (240 BPM = 0.25s apart)
    let fast_onsets: Vec<f32> = (0..20).map(|i| i as f32 * 0.25).collect();
    let fast_tempo = tracker.estimate_tempo(&fast_onsets);

    // Tracker has max tempo of 300 BPM; 240 BPM is within range
    // but octave ambiguity check may fold it to 120 BPM
    if let Some(tempo) = fast_tempo {
        assert!(tempo <= 300.0, "Should respect max tempo limit");
    }

    // Very slow tempo (20 BPM = 3s apart) — below min of 30 BPM
    let slow_onsets = vec![0.0, 3.0, 6.0, 9.0, 12.0];
    let slow_tempo = tracker.estimate_tempo(&slow_onsets);

    // Tracker has min tempo of 30 BPM, so 20 BPM might not be detected
    // or might be detected as double-time
    if let Some(tempo) = slow_tempo {
        assert!(tempo >= 30.0, "Should respect min tempo limit");
    }
}
