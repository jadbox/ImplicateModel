//! `implicate-rolpa`: The `@rolpa` Shadow-Manifold Projector and 1-Pass `CliffordNet-Lite`
//! GPU Visual Refiner (Subsystem IV, Track R3.3).

pub mod acoustic;
pub mod cliffordnet;
pub mod gpu_refiner;

pub use acoustic::{
    AcousticRolpaProjector, DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS, ModalResonatorVoice,
    estimate_fundamental_frequency_hz,
};
pub use cliffordnet::{
    CLIFFORDNET_BLOCKS, CLIFFORDNET_SHIFTS, CliffordNetLiteRefiner, RolpaInputGrid, RolpaUniforms,
    compute_lpips_rgb, compute_psnr_rgb, compute_temporal_warping_error,
    simulate_raw_wavepacket_splat,
};
pub use gpu_refiner::{
    GpuCliffordNetRefiner, GpuModelSheetUnfolder, GpuPartPose, RolpaGpuContext, UnfoldUniforms,
};
