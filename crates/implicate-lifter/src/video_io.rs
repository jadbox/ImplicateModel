//! `ffmpeg`-backed video IO: raw RGB24 frame reader (native and scaled),
//! H.264 writer, and PNG/PPM capture.
//!
//! Evolved from `holonengine/crates/cc_field/src/video_io.rs` with zero-allocation
//! streaming buffers (`next_frame_into`) and hardware-threaded scaled analysis pipes
//! (`open_range_scaled`) for $> 180\text{ FPS}$ 1080p shot segmentation.

use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Stream metadata from `ffprobe`.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    /// Frames per second (rational converted to `f64`).
    pub fps: f64,
    /// Frame count from the container, `0` when unknown.
    pub frames: u64,
}

impl VideoInfo {
    pub fn frame_bytes(&self) -> usize {
        self.width as usize * self.height as usize * 3
    }
}

fn run_capture(cmd: &mut Command) -> io::Result<String> {
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "command failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Probe stream `width,height,fps,frames`.
pub fn probe(path: &Path) -> io::Result<VideoInfo> {
    let out = run_capture(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_frames",
                "-of",
                "csv=p=0",
            ])
            .arg(path),
    )?;
    let mut it = out.split(',');
    let width: u32 = it
        .next()
        .ok_or_else(|| io::Error::other("ffprobe: missing width"))?
        .trim()
        .parse()
        .map_err(|_| io::Error::other("ffprobe: bad width"))?;
    let height: u32 = it
        .next()
        .ok_or_else(|| io::Error::other("ffprobe: missing height"))?
        .trim()
        .parse()
        .map_err(|_| io::Error::other("ffprobe: bad height"))?;
    let rate = it.next().unwrap_or("30/1").trim();
    let fps = match rate.split_once('/') {
        Some((n, d)) => n.parse::<f64>().unwrap_or(30.0) / d.parse::<f64>().unwrap_or(1.0),
        None => rate.parse().unwrap_or(30.0),
    };
    let frames: u64 = it.next().unwrap_or("0").trim().parse().unwrap_or(0);
    Ok(VideoInfo {
        width,
        height,
        fps,
        frames,
    })
}

/// One decoded RGB24 frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl RawFrame {
    pub fn new(width: u32, height: u32, rgb: Vec<u8>) -> Self {
        debug_assert_eq!(rgb.len(), width as usize * height as usize * 3);
        Self { width, height, rgb }
    }

    pub fn blank(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgb: vec![0u8; width as usize * height as usize * 3],
        }
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let i = (y as usize * self.width as usize + x as usize) * 3;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }
}

/// Streaming raw RGB24 reader over a span of an ffmpeg-decodable video.
pub struct FrameReader {
    child: Child,
    stdout: BufReader<ChildStdout>,
    width: u32,
    height: u32,
    buf: Vec<u8>,
    done: bool,
}

impl FrameReader {
    /// Open the whole stream at native resolution.
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_range(path, None, None)
    }

    /// Open `path` at native resolution, seeking to `start` seconds and reading at most `duration`.
    pub fn open_range(path: &Path, start: Option<f64>, duration: Option<f64>) -> io::Result<Self> {
        let info = probe(path)?;
        Self::open_range_with_scale(path, start, duration, info.width, info.height, false)
    }

    /// Open `path` scaled to `(target_w, target_h)` using `fast_bilinear` for high-throughput
    /// streaming shot segmentation and multi-plane optical flow analysis.
    pub fn open_range_scaled(
        path: &Path,
        start: Option<f64>,
        duration: Option<f64>,
        target_w: u32,
        target_h: u32,
    ) -> io::Result<Self> {
        let info = probe(path)?;
        let scaled = info.width != target_w || info.height != target_h;
        Self::open_range_with_scale(path, start, duration, target_w, target_h, scaled)
    }

    fn open_range_with_scale(
        path: &Path,
        start: Option<f64>,
        duration: Option<f64>,
        width: u32,
        height: u32,
        scale_filter: bool,
    ) -> io::Result<Self> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-threads", "0"]);
        if let Some(s) = start {
            cmd.arg("-ss").arg(format!("{s:.6}"));
        }
        cmd.arg("-i").arg(path);
        if let Some(d) = duration {
            cmd.arg("-t").arg(format!("{d:.6}"));
        }
        if scale_filter {
            cmd.arg("-vf")
                .arg(format!("scale={width}:{height}:flags=fast_bilinear"));
        }
        let frame_bytes = width as usize * height as usize * 3;
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null());
        let mut child = cmd.spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("ffmpeg: no stdout"))?;
        Ok(Self {
            child,
            stdout: BufReader::with_capacity(frame_bytes.max(65536), stdout),
            width,
            height,
            buf: vec![0u8; frame_bytes],
            done: false,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Read the next frame into an existing `RawFrame` buffer without heap allocation.
    pub fn next_frame_into(&mut self, out: &mut RawFrame) -> io::Result<bool> {
        if self.done {
            return Ok(false);
        }
        let frame_bytes = self.width as usize * self.height as usize * 3;
        if out.rgb.len() != frame_bytes {
            out.rgb.resize(frame_bytes, 0);
        }
        out.width = self.width;
        out.height = self.height;
        match read_exact_or_eof(&mut self.stdout, &mut out.rgb)? {
            false => {
                self.done = true;
                Ok(false)
            }
            true => Ok(true),
        }
    }

    /// Read the next frame, `None` at end of stream.
    pub fn next_frame(&mut self) -> io::Result<Option<RawFrame>> {
        if self.done {
            return Ok(None);
        }
        match read_exact_or_eof(&mut self.stdout, &mut self.buf)? {
            false => {
                self.done = true;
                Ok(None)
            }
            true => Ok(Some(RawFrame {
                width: self.width,
                height: self.height,
                rgb: self.buf.clone(),
            })),
        }
    }

    /// Sweep the whole span, yielding frames to `f`.
    pub fn for_each<F>(&mut self, mut f: F) -> io::Result<usize>
    where
        F: FnMut(usize, &RawFrame),
    {
        let mut n = 0;
        let mut frame = RawFrame::blank(self.width, self.height);
        while self.next_frame_into(&mut frame)? {
            f(n, &frame);
            n += 1;
        }
        Ok(n)
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn read_exact_or_eof(r: &mut impl Read, buf: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => {
                if filled == 0 {
                    return Ok(false);
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated frame",
                ));
            }
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// H.264 MP4 writer fed with raw RGB24 frames.
pub struct VideoWriter {
    child: Child,
    stdin: Option<ChildStdin>,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    written: u64,
}

impl VideoWriter {
    pub fn create(path: &Path, width: u32, height: u32, fps: f64) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut child = Command::new("ffmpeg")
            .arg("-y")
            .arg("-v")
            .arg("error")
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .arg("-s")
            .arg(format!("{width}x{height}"))
            .arg("-r")
            .arg(format!("{fps}"))
            .arg("-i")
            .arg("-")
            .arg("-an")
            .args([
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ])
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("ffmpeg: no stdin"))?;
        Ok(Self {
            child,
            stdin: Some(stdin),
            width,
            height,
            fps,
            written: 0,
        })
    }

    pub fn write_frame(&mut self, rgb: &[u8]) -> io::Result<()> {
        let expect = self.width as usize * self.height as usize * 3;
        if rgb.len() != expect {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("frame has {} bytes, expected {expect}", rgb.len()),
            ));
        }
        self.stdin
            .as_mut()
            .ok_or_else(|| io::Error::other("writer already finished"))?
            .write_all(rgb)?;
        self.written += 1;
        Ok(())
    }

    pub fn written(&self) -> u64 {
        self.written
    }

    /// Close stdin and wait for the encoder.
    pub fn finish(&mut self) -> io::Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        if !status.success() {
            return Err(io::Error::other("ffmpeg encoder failed"));
        }
        Ok(())
    }
}

impl Drop for VideoWriter {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Write a binary PPM (`P6`).
pub fn write_ppm(path: &Path, width: u32, height: u32, rgb: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    write!(f, "P6\n{width} {height}\n255\n")?;
    f.write_all(rgb)?;
    f.flush()
}

/// Write an RGB PNG via the `image` crate (no ffmpeg round-trip).
pub fn write_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let img = image::RgbImage::from_raw(width, height, rgb.to_vec())
        .ok_or_else(|| io::Error::other("buffer size does not match dimensions"))?;
    img.save_with_format(path, image::ImageFormat::Png)
        .map_err(|e| io::Error::other(e.to_string()))
}

/// Run one `ffmpeg -ss t -frames:v 1` extraction to PNG.
pub fn extract_frame_png(video: &Path, t: f64, out: &Path) -> io::Result<()> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .arg("-ss")
        .arg(format!("{t:.6}"))
        .arg("-i")
        .arg(video)
        .args(["-frames:v", "1"])
        .arg(out)
        .status()?;
    if !status.success() {
        return Err(io::Error::other("ffmpeg frame extraction failed"));
    }
    Ok(())
}

/// Read all frames in `[start, start + duration]` at native resolution into memory.
pub fn read_frames(
    path: &Path,
    start: Option<f64>,
    duration: Option<f64>,
) -> io::Result<Vec<RawFrame>> {
    let mut reader = FrameReader::open_range(path, start, duration)?;
    let mut out = Vec::new();
    while let Some(f) = reader.next_frame()? {
        out.push(f);
    }
    Ok(out)
}

/// Read all frames in `[start, start + duration]` scaled to `(target_w, target_h)`.
pub fn read_frames_scaled(
    path: &Path,
    start: Option<f64>,
    duration: Option<f64>,
    target_w: u32,
    target_h: u32,
) -> io::Result<Vec<RawFrame>> {
    let mut reader = FrameReader::open_range_scaled(path, start, duration, target_w, target_h)?;
    let mut out = Vec::new();
    while let Some(f) = reader.next_frame()? {
        out.push(f);
    }
    Ok(out)
}

/// Default output directory for diagnostics (`/tmp/opencode`).
pub fn scratch_dir() -> PathBuf {
    let base = std::env::var_os("TMPDIR").unwrap_or_else(|| "/tmp".into());
    PathBuf::from(base).join("opencode")
}

/// Mux raw RGB24 frames and a synchronized `.wav` stereo audio file into a playable H.264 + AAC `.mp4`.
pub fn mux_rgb_frames_and_wav_to_mp4(
    mp4_path: &Path,
    width: u32,
    height: u32,
    fps: f64,
    rgb_frames: &[Vec<u8>],
    wav_path: &Path,
) -> io::Result<()> {
    if let Some(parent) = mp4_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let expect = (width as usize) * (height as usize) * 3;
    let mut child = Command::new("ffmpeg")
        .arg("-y")
        .arg("-v")
        .arg("error")
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg("-s")
        .arg(format!("{width}x{height}"))
        .arg("-r")
        .arg(format!("{fps}"))
        .arg("-i")
        .arg("-")
        .arg("-i")
        .arg(wav_path)
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
        ])
        .arg(mp4_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::other("ffmpeg muxer: no stdin"))?;
        for frame in rgb_frames {
            if frame.len() != expect {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("frame has {} bytes, expected {expect}", frame.len()),
                ));
            }
            stdin.write_all(frame)?;
        }
    }
    drop(child.stdin.take());
    let status = child.wait()?;
    if !status.success() {
        return Err(io::Error::other("ffmpeg MP4 audio+video muxer failed"));
    }
    Ok(())
}

/// Probe primary audio stream `(sample_rate_hz, channels)` from a media container via `ffprobe`.
pub fn probe_audio_stream(path: &Path) -> io::Result<(u32, u32)> {
    let out = run_capture(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "csv=p=0",
            ])
            .arg(path),
    )?;
    let mut it = out.split(',');
    let sr: u32 = it
        .next()
        .ok_or_else(|| io::Error::other("ffprobe audio: missing sample_rate"))?
        .trim()
        .parse()
        .map_err(|_| io::Error::other("ffprobe audio: bad sample_rate"))?;
    let ch: u32 = it
        .next()
        .ok_or_else(|| io::Error::other("ffprobe audio: missing channels"))?
        .trim()
        .parse()
        .map_err(|_| io::Error::other("ffprobe audio: bad channels"))?;
    Ok((sr, ch))
}
