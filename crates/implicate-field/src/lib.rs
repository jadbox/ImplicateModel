//! `implicate-field`: Unified Staggered Yee FDTD Wave-Matter Lattice, Chromatic Luminescence,
//! and Articulated `CliffordDensityElement` Holon Coupling (Subsystem III, Track R4.1).

pub mod gpu_lattice;
pub mod holon_coupling;

pub use gpu_lattice::{
    FieldGpuContext, GpuWaveMatterLattice, LatticeParams, MAX_GPU_HOLONS,
};
pub use holon_coupling::{
    ArticulatedFieldHolon, CollisionImpulseEvent, GpuHolonQuery, GpuHolonReduction, GpuHolonSource,
    SubHolonLink,
};
