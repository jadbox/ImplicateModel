//! Synchronized Acoustic `@rolpa` Projector: $48\text{ kHz}$ Modal & Harmonic Stereo Synthesis
//! (Track R4.2, Whitepaper §7.2).
//!
//! Couples PGA collision impulses ($\Delta \mathbf{P}_B$ at $P_i \wedge B_k = 0$) and continuous
//! binaural camera wave pressures $(p_L, p_R)$ into a sample-accurate $48\text{ kHz}$ stereo stream:
//! 1. **Collision Modal Resonators:** Impacts excite a 4-mode damped resonator bank tuned to the
//!    struck holon's constitutive wave speed $f_0 = 440\sqrt{K/\rho}$ Hz at the exact sub-frame
//!    sample index of contact ($< 1\text{ ms}$ audio-visual synchronization).
//! 2. **Continuous Binaural Wake Sonification:** Port and starboard camera pickups $(p_L, p_R)$
//!    are linearly interpolated across the 800-sample frame block (`48000 Hz / 60 FPS`), filtered
//!    through a 1st-order DC blocker, and mixed with a continuous Doppler carrier proportional
//!    to the wave pressure envelope.

use std::collections::VecDeque;
use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use implicate_field::CollisionImpulseEvent;
use serde::{Deserialize, Serialize};

pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;
pub const DEFAULT_VIDEO_FPS: u32 = 60;
const MODAL_RATIOS: [f64; 4] = [1.0, 2.0, 2.756, 3.860];
const MODAL_WEIGHTS: [f32; 4] = [0.52, 0.24, 0.15, 0.09];
const MODAL_DECAYS: [f32; 4] = [14.0, 22.0, 34.0, 48.0];

/// A 4-partial damped modal resonator voice triggered by a PGA `CollisionImpulseEvent`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModalResonatorVoice {
    pub holon_id: u32,
    pub fundamental_hz: f32,
    pub amplitude: f32,
    pub pan_left: f32,
    pub pan_right: f32,
    pub phases: [f64; 4],
    pub elapsed_sec: f32,
    /// Remaining sub-frame delay in samples before the impact onset within the current frame block.
    pub onset_delay_samples: usize,
}

impl ModalResonatorVoice {
    pub fn from_collision(
        event: &CollisionImpulseEvent,
        frame_start_time: f64,
        sample_rate: u32,
        camera_center_x: f32,
        camera_half_width: f32,
    ) -> Self {
        let fundamental_hz = (440.0 * event.wave_speed_c).clamp(80.0, 6000.0);
        let amplitude = (event.impulse_magnitude * 0.25).clamp(0.08, 0.95);
        // Equal-power stereo panning from spatial contact coordinate relative to camera
        let rel_x =
            ((event.contact_pos[0] - camera_center_x) / camera_half_width.max(1.0)).clamp(-1.0, 1.0);
        let pan_angle = (rel_x + 1.0) * (std::f32::consts::FRAC_PI_4);
        let pan_left = pan_angle.cos();
        let pan_right = pan_angle.sin();

        let dt_into_frame = (event.sim_time - frame_start_time).max(0.0);
        let onset_delay_samples = (dt_into_frame * (sample_rate as f64)).round() as usize;

        Self {
            holon_id: event.holon_id,
            fundamental_hz,
            amplitude,
            pan_left,
            pan_right,
            phases: [0.0; 4],
            elapsed_sec: 0.0,
            onset_delay_samples,
        }
    }

    /// Advance the 4-mode resonator by one audio sample (`dt = 1 / sample_rate`).
    #[inline]
    pub fn next_sample(&mut self, inv_sr: f32) -> [f32; 2] {
        if self.onset_delay_samples > 0 {
            self.onset_delay_samples -= 1;
            return [0.0, 0.0];
        }

        let two_pi = 2.0 * std::f64::consts::PI;
        let f0 = self.fundamental_hz as f64;
        let dt_f64 = inv_sr as f64;
        // Fast percussive attack (0.15 ms) + modal exponential decay
        let attack = 1.0 - (-self.elapsed_sec / 0.00015).exp();
        let mut mono = 0.0f32;

        for m in 0..4 {
            let env = (-MODAL_DECAYS[m] * self.elapsed_sec).exp();
            let s = (self.phases[m].cos() as f32) * MODAL_WEIGHTS[m] * env;
            mono += s;
            self.phases[m] = (self.phases[m] + two_pi * f0 * MODAL_RATIOS[m] * dt_f64) % two_pi;
        }

        self.elapsed_sec += inv_sr;
        let val = self.amplitude * (0.25 + 0.75 * attack) * mono;
        [val * self.pan_left, val * self.pan_right]
    }

    #[inline]
    pub fn is_finished(&self) -> bool {
        self.elapsed_sec > 0.35 || (self.amplitude * (-MODAL_DECAYS[0] * self.elapsed_sec).exp() < 1e-4)
    }
}

/// Synchronized $48\text{ kHz}$ stereo acoustic `@rolpa` projector.
#[derive(Clone, Debug)]
pub struct AcousticRolpaProjector {
    pub sample_rate: u32,
    pub video_fps: u32,
    pub samples_per_frame: usize,
    pub active_voices: Vec<ModalResonatorVoice>,
    pub ring_buffer: VecDeque<[f32; 2]>,
    pub max_ring_samples: usize,
    pub underrun_count: usize,
    prev_ear_p: [f32; 2],
    dc_prev_in: [f32; 2],
    dc_prev_out: [f32; 2],
    wake_carrier_phase: f64,
}

impl Default for AcousticRolpaProjector {
    fn default() -> Self {
        Self::new(DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS)
    }
}

impl AcousticRolpaProjector {
    pub fn new(sample_rate: u32, video_fps: u32) -> Self {
        let samples_per_frame = (sample_rate / video_fps.max(1)) as usize;
        let max_ring_samples = (sample_rate as usize) * 2;
        Self {
            sample_rate,
            video_fps,
            samples_per_frame,
            active_voices: Vec::new(),
            ring_buffer: VecDeque::with_capacity(max_ring_samples),
            max_ring_samples,
            underrun_count: 0,
            prev_ear_p: [0.0, 0.0],
            dc_prev_in: [0.0, 0.0],
            dc_prev_out: [0.0, 0.0],
            wake_carrier_phase: 0.0,
        }
    }

    /// Synthesize one video frame's worth of $48\text{ kHz}$ stereo samples (`800` samples at $60\text{ FPS}$)
    /// from the camera binaural pressures `ear_p = [p_left, p_right]` and any PGA `CollisionImpulseEvent`s
    /// occurring during this frame.
    pub fn synthesize_frame(
        &mut self,
        frame_start_time: f64,
        ear_p: [f32; 2],
        collisions: &[CollisionImpulseEvent],
        camera_center_x: f32,
        camera_half_width: f32,
    ) -> Vec<[f32; 2]> {
        for ev in collisions {
            self.active_voices.push(ModalResonatorVoice::from_collision(
                ev,
                frame_start_time,
                self.sample_rate,
                camera_center_x,
                camera_half_width,
            ));
        }

        let n = self.samples_per_frame;
        let inv_sr = 1.0 / (self.sample_rate as f32);
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut frame_audio = Vec::with_capacity(n);

        for s_idx in 0..n {
            let alpha = (s_idx as f32) / (n.max(1) as f32);
            let raw_l = self.prev_ear_p[0] + alpha * (ear_p[0] - self.prev_ear_p[0]);
            let raw_r = self.prev_ear_p[1] + alpha * (ear_p[1] - self.prev_ear_p[1]);

            // 1st-order DC blocking filter on continuous lattice pressure
            const R_DC: f32 = 0.995;
            let dc_l = raw_l - self.dc_prev_in[0] + R_DC * self.dc_prev_out[0];
            self.dc_prev_in[0] = raw_l;
            self.dc_prev_out[0] = dc_l;

            let dc_r = raw_r - self.dc_prev_in[1] + R_DC * self.dc_prev_out[1];
            self.dc_prev_in[1] = raw_r;
            self.dc_prev_out[1] = dc_r;

            // Continuous aerodynamic/wake harmonic modulated by local pressure magnitude
            let wake_env_l = raw_l.abs().min(1.5) * 0.08;
            let wake_env_r = raw_r.abs().min(1.5) * 0.08;
            let carrier_hz = 110.0 + 60.0 * ((raw_l.abs() + raw_r.abs()) as f64).min(2.0);
            let carrier = self.wake_carrier_phase.sin() as f32;
            self.wake_carrier_phase =
                (self.wake_carrier_phase + two_pi * carrier_hz * (inv_sr as f64)) % two_pi;

            let mut out_l = 0.25 * dc_l + wake_env_l * carrier;
            let mut out_r = 0.25 * dc_r + wake_env_r * carrier;

            for voice in &mut self.active_voices {
                let [vl, vr] = voice.next_sample(inv_sr);
                out_l += vl;
                out_r += vr;
            }

            let sample = [out_l.tanh(), out_r.tanh()];
            frame_audio.push(sample);
            if self.ring_buffer.len() >= self.max_ring_samples {
                self.ring_buffer.pop_front();
            }
            self.ring_buffer.push_back(sample);
        }

        self.prev_ear_p = ear_p;
        self.active_voices.retain(|v| !v.is_finished());
        frame_audio
    }

    /// Consume `num_samples` stereo frames from the real-time output ring buffer.
    /// Increments `self.underrun_count` if fewer than `num_samples` are available.
    pub fn consume_audio_block(&mut self, num_samples: usize) -> Vec<[f32; 2]> {
        if self.ring_buffer.len() < num_samples {
            self.underrun_count += 1;
        }
        let mut out = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            out.push(self.ring_buffer.pop_front().unwrap_or([0.0, 0.0]));
        }
        out
    }

    /// Export a slice of stereo `[left, right]` samples to a 16-bit PCM WAV file.
    pub fn write_stereo_wav(
        &self,
        path: &Path,
        samples: &[[f32; 2]],
    ) -> Result<(), hound::Error> {
        let spec = WavSpec {
            channels: 2,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)?;
        for &[l, r] in samples {
            let il = (l.clamp(-1.0, 1.0) * 32767.0) as i16;
            let ir = (r.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(il)?;
            writer.write_sample(ir)?;
        }
        writer.finalize()
    }
}

/// Estimate the dominant fundamental frequency (in Hz) of a mono or stereo signal window
/// via parabolic-interpolated autocorrelation peak detection.
pub fn estimate_fundamental_frequency_hz(samples: &[[f32; 2]], sample_rate: u32) -> f32 {
    let n = samples.len();
    if n < 64 {
        return 0.0;
    }
    let mono: Vec<f64> = samples.iter().map(|s| (s[0] + s[1]) as f64).collect();
    let min_lag = (sample_rate as usize / 2500).max(4);
    let max_lag = (sample_rate as usize / 80).min(n / 2);

    let mut best_lag = min_lag;
    let mut best_corr = f64::NEG_INFINITY;
    let mut corr_vec = vec![0.0f64; max_lag + 2];

    for lag in min_lag..=max_lag {
        let mut sum = 0.0f64;
        for i in 0..(n - lag) {
            sum += mono[i] * mono[i + lag];
        }
        corr_vec[lag] = sum;
    }

    // Find first prominent local maximum after initial zero-crossing
    let mut crossed_negative = false;
    for lag in min_lag..max_lag {
        if corr_vec[lag] < 0.0 {
            crossed_negative = true;
        }
        if crossed_negative
            && corr_vec[lag] > corr_vec[lag - 1]
            && corr_vec[lag] >= corr_vec[lag + 1]
            && corr_vec[lag] > best_corr
        {
            best_corr = corr_vec[lag];
            best_lag = lag;
        }
    }

    let c_prev = corr_vec[best_lag.saturating_sub(1)];
    let c_curr = corr_vec[best_lag];
    let c_next = corr_vec[best_lag + 1];
    let denom = c_prev - 2.0 * c_curr + c_next;
    let delta = if denom.abs() > 1e-9 {
        0.5 * (c_prev - c_next) / denom
    } else {
        0.0
    };
    let refined_lag = (best_lag as f64 + delta.clamp(-0.5, 0.5)).max(1.0);
    (sample_rate as f64 / refined_lag) as f32
}
