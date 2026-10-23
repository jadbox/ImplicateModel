//! Hardware `wgpu` Staggered Yee FDTD Wave-Matter Lattice + Chromatic Luminescence +
//! Zero-Readback Holon & Binaural Probe Reduction (Track R4.1, Whitepaper §6.2 & §6.3).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use bytemuck::{Pod, Zeroable};
use implicate_math::OrientedLine2;
use serde::{Deserialize, Serialize};
use wgpu::*;

use crate::holon_coupling::{
    ArticulatedFieldHolon, CollisionImpulseEvent, GpuHolonQuery, GpuHolonReduction, GpuHolonSource,
};

pub const MAX_GPU_HOLONS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct LatticeParams {
    pub width: u32,
    pub height: u32,
    pub num_sources: u32,
    pub num_queries: u32,
    pub dx: f32,
    pub dy: f32,
    pub dt: f32,
    pub inv_dx: f32,
    pub inv_dy: f32,
    pub beta: f32,
    pub alpha: f32,
    pub threshold_decay: f32,
    pub kerr_lensing: f32,
    pub bohmian_gain: f32,
    pub outer_periodic: u32,
    pub _pad0: u32,
}

impl Default for LatticeParams {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            num_sources: 0,
            num_queries: 0,
            dx: 1.0,
            dy: 1.0,
            dt: 0.15,
            inv_dx: 1.0,
            inv_dy: 1.0,
            beta: 0.35,
            alpha: 0.25,
            threshold_decay: 0.08,
            kerr_lensing: 0.22,
            bohmian_gain: 0.035,
            outer_periodic: 0,
            _pad0: 0,
        }
    }
}

#[derive(Clone)]
pub struct FieldGpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

static SHARED_FIELD_GPU: OnceLock<FieldGpuContext> = OnceLock::new();

impl FieldGpuContext {
    pub fn shared() -> &'static Self {
        SHARED_FIELD_GPU.get_or_init(|| {
            pollster::block_on(Self::init_headless())
                .expect("implicate-field requires a WGPU-capable GPU adapter")
        })
    }

    async fn init_headless() -> Result<Self, String> {
        let instance = Instance::default();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Failed to request WGPU adapter: {e}"))?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("implicate-field Wave-Matter GPU Device"),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Failed to request WGPU device: {e}"))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

/// Zero-readback GPU Wave-Matter Lattice with ping-pong storage buffers (`state_a/b`, `lum_a/b`),
/// constitutive material buffer (`rho, K, gamma, cohesion_C`), and compact holon/probe reduction.
pub struct GpuWaveMatterLattice {
    pub params: LatticeParams,
    pub uniform_buffer: Buffer,
    pub state_a: Buffer,
    pub state_b: Buffer,
    pub material_buf: Buffer,
    pub lum_a: Buffer,
    pub lum_b: Buffer,
    pub sources_buf: Buffer,
    pub queries_buf: Buffer,
    pub reductions_buf: Buffer,
    pub reductions_staging: Buffer,
    pub field_staging: Buffer,
    pub vel_pipeline: ComputePipeline,
    pub press_lum_pipeline: ComputePipeline,
    pub reduce_pipeline: ComputePipeline,
    pub bind_groups: [BindGroup; 2],
    pub parity: AtomicUsize,
}

impl GpuWaveMatterLattice {
    pub fn new(ctx: &FieldGpuContext, params: LatticeParams) -> Self {
        let num_cells = (params.width as usize) * (params.height as usize);
        let grid_byte_size = (num_cells * 16) as u64;

        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("implicate-field lattice.wgsl"),
            source: ShaderSource::Wgsl(include_str!("lattice.wgsl").into()),
        });

        let uniform_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("LatticeParams Uniform"),
            size: std::mem::size_of::<LatticeParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&params));

        let make_grid_buf = |label: &str, init_val: [f32; 4]| -> Buffer {
            let buf = ctx.device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: grid_byte_size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let init_vec = vec![init_val; num_cells];
            ctx.queue
                .write_buffer(&buf, 0, bytemuck::cast_slice(&init_vec));
            buf
        };

        let state_a = make_grid_buf("lattice_state_a", [0.0, 0.0, 0.0, 0.0]);
        let state_b = make_grid_buf("lattice_state_b", [0.0, 0.0, 0.0, 0.0]);
        // Default constitutive medium: rho = 1.0, K = 1.0, gamma = 0.03, cohesion = 1.0
        let material_buf = make_grid_buf("lattice_material", [1.0, 1.0, 0.03, 1.0]);
        let lum_a = make_grid_buf("lattice_lum_a", [0.0, 0.0, 0.0, 0.85]);
        let lum_b = make_grid_buf("lattice_lum_b", [0.0, 0.0, 0.0, 0.85]);

        let sources_buf = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("lattice_holon_sources"),
            size: (MAX_GPU_HOLONS * std::mem::size_of::<GpuHolonSource>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let queries_buf = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("lattice_holon_queries"),
            size: (MAX_GPU_HOLONS * std::mem::size_of::<GpuHolonQuery>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let red_bytes = (MAX_GPU_HOLONS * std::mem::size_of::<GpuHolonReduction>()) as u64;
        let reductions_buf = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("lattice_holon_reductions"),
            size: red_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let reductions_staging = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("lattice_reductions_staging"),
            size: red_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let field_staging = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("lattice_field_staging"),
            size: grid_byte_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl_entry = |binding: u32, read_only: bool| -> BindGroupLayoutEntry {
            BindGroupLayoutEntry {
                binding,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };

        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Wave-Matter Lattice BGL"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    bgl_entry(1, true),  // state_in
                    bgl_entry(2, false), // state_out
                    bgl_entry(3, false), // material
                    bgl_entry(4, true),  // lum_in
                    bgl_entry(5, false), // lum_out
                    bgl_entry(6, true),  // sources
                    bgl_entry(7, true),  // queries
                    bgl_entry(8, false), // reductions
                ],
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Wave-Matter Lattice PipelineLayout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let make_pipe = |entry: &str| -> ComputePipeline {
            ctx.device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                })
        };

        let vel_pipeline = make_pipe("step_velocity");
        let press_lum_pipeline = make_pipe("step_pressure_and_luminescence");
        let reduce_pipeline = make_pipe("reduce_holons");

        let make_bg = |s_in: &Buffer, s_out: &Buffer, l_in: &Buffer, l_out: &Buffer| -> BindGroup {
            ctx.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Wave-Matter PingPong BG"),
                layout: &bgl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: s_in.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: s_out.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: material_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: l_in.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: l_out.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: sources_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: queries_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 8,
                        resource: reductions_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let bg_ab = make_bg(&state_a, &state_b, &lum_a, &lum_b);
        let bg_ba = make_bg(&state_b, &state_a, &lum_b, &lum_a);

        Self {
            params,
            uniform_buffer,
            state_a,
            state_b,
            material_buf,
            lum_a,
            lum_b,
            sources_buf,
            queries_buf,
            reductions_buf,
            reductions_staging,
            field_staging,
            vel_pipeline,
            press_lum_pipeline,
            reduce_pipeline,
            bind_groups: [bg_ab, bg_ba],
            parity: AtomicUsize::new(0),
        }
    }

    /// Execute `substeps` Yee wave-matter updates + zero-readback GPU holon/probe reduction
    /// in a single command submission, returning the compact `GpuHolonReduction` records.
    pub fn step_and_reduce_gpu(
        &mut self,
        ctx: &FieldGpuContext,
        sources: &[GpuHolonSource],
        queries: &[GpuHolonQuery],
        substeps: usize,
    ) -> Vec<GpuHolonReduction> {
        let n_src = sources.len().min(MAX_GPU_HOLONS);
        let n_qry = queries.len().min(MAX_GPU_HOLONS);
        self.params.num_sources = n_src as u32;
        self.params.num_queries = n_qry as u32;

        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.params));
        if n_src > 0 {
            ctx.queue.write_buffer(
                &self.sources_buf,
                0,
                bytemuck::cast_slice(&sources[..n_src]),
            );
        }
        if n_qry > 0 {
            ctx.queue.write_buffer(
                &self.queries_buf,
                0,
                bytemuck::cast_slice(&queries[..n_qry]),
            );
        }

        let wg_x = (self.params.width + 15) / 16;
        let wg_y = (self.params.height + 15) / 16;
        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Wave-Matter Coupled Step & Reduce"),
        });

        let mut last_bg_idx = self.parity.load(Ordering::Relaxed);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Yee Wave-Matter + Reduce Pass"),
                timestamp_writes: None,
            });
            for _ in 0..substeps.max(1) {
                let bg = &self.bind_groups[last_bg_idx];
                pass.set_bind_group(0, bg, &[]);
                pass.set_pipeline(&self.vel_pipeline);
                pass.dispatch_workgroups(wg_x, wg_y, 1);

                pass.set_pipeline(&self.press_lum_pipeline);
                pass.dispatch_workgroups(wg_x, wg_y, 1);

                last_bg_idx ^= 1;
            }
            // Reduce holons on the most recently written state_out / lum_out
            let active_bg = &self.bind_groups[last_bg_idx ^ 1];
            if n_qry > 0 {
                pass.set_bind_group(0, active_bg, &[]);
                pass.set_pipeline(&self.reduce_pipeline);
                let wg_q = ((n_qry as u32) + 31) / 32;
                pass.dispatch_workgroups(wg_q, 1, 1);
            }
        }
        self.parity.store(last_bg_idx, Ordering::Relaxed);

        if n_qry == 0 {
            ctx.queue.submit(std::iter::once(encoder.finish()));
            return Vec::new();
        }

        let red_byte_len = (n_qry * std::mem::size_of::<GpuHolonReduction>()) as u64;
        encoder.copy_buffer_to_buffer(
            &self.reductions_buf,
            0,
            &self.reductions_staging,
            0,
            red_byte_len,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.reductions_staging.slice(0..red_byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let out = {
            let view = slice.get_mapped_range().unwrap();
            let recs: &[GpuHolonReduction] = bytemuck::cast_slice(&view);
            recs.to_vec()
        };
        self.reductions_staging.unmap();
        out
    }

    /// Execute one complete coupled macro-frame (`substeps` GPU wave-matter steps + zero-readback
    /// reduction + articulated `CliffordDensityElement` Holon integration + binaural camera ear probes).
    #[allow(clippy::too_many_arguments)]
    pub fn step_coupled_frame(
        &mut self,
        ctx: &FieldGpuContext,
        holons: &mut [ArticulatedFieldHolon],
        external_accels: &[[f32; 2]],
        boundaries: &[OrientedLine2],
        camera_ear_left: [f32; 2],
        camera_ear_right: [f32; 2],
        frame_idx: u64,
        sim_time: f64,
        substeps: usize,
    ) -> ([f32; 2], Vec<CollisionImpulseEvent>) {
        let mut sources = Vec::with_capacity(holons.len());
        let mut queries = Vec::with_capacity(holons.len() + 2);
        for h in holons.iter() {
            sources.push(h.to_gpu_source());
            queries.push(h.to_gpu_query());
        }
        // Append 2 point probes for port (left) and starboard (right) camera binaural pickups
        queries.push(GpuHolonQuery {
            x: camera_ear_left[0],
            y: camera_ear_left[1],
            radius: 0.0,
            yield_pressure: 1.0,
        });
        queries.push(GpuHolonQuery {
            x: camera_ear_right[0],
            y: camera_ear_right[1],
            radius: 0.0,
            yield_pressure: 1.0,
        });

        let reductions = self.step_and_reduce_gpu(ctx, &sources, &queries, substeps);
        let mut events = Vec::new();
        let bounds_wh = [self.params.width as f32, self.params.height as f32];
        let macro_dt = self.params.dt * (substeps.max(1) as f32);

        for (i, holon) in holons.iter_mut().enumerate() {
            let red = reductions.get(i).copied().unwrap_or_default();
            let ext_a = external_accels.get(i).copied().unwrap_or([0.0, 0.0]);
            holon.integrate_step(
                red,
                ext_a,
                boundaries,
                bounds_wh,
                sim_time,
                frame_idx,
                0,
                macro_dt,
                &mut events,
            );
        }

        let p_left = reductions
            .get(holons.len())
            .map(|r| r.forces[3])
            .unwrap_or(0.0);
        let p_right = reductions
            .get(holons.len() + 1)
            .map(|r| r.forces[3])
            .unwrap_or(0.0);

        ([p_left, p_right], events)
    }

    /// Read back the current `(p, vx, vy, Q)` state grid (only used in tests/diagnostics).
    pub fn readback_state_grid(&self, ctx: &FieldGpuContext) -> Vec<[f32; 4]> {
        let active_parity = self.parity.load(Ordering::Relaxed);
        // After step_and_reduce_gpu flips parity, the most recently written state_out is state_b when parity==1, or state_a when parity==0
        let src_buf = if active_parity == 1 {
            &self.state_b
        } else {
            &self.state_a
        };
        self.readback_grid_buffer(ctx, src_buf)
    }

    /// Read back the current `(L_r, L_g, L_b, p_yield)` luminescence grid (only used in tests/diagnostics).
    pub fn readback_luminescence_grid(&self, ctx: &FieldGpuContext) -> Vec<[f32; 4]> {
        let active_parity = self.parity.load(Ordering::Relaxed);
        let src_buf = if active_parity == 1 {
            &self.lum_b
        } else {
            &self.lum_a
        };
        self.readback_grid_buffer(ctx, src_buf)
    }

    fn readback_grid_buffer(&self, ctx: &FieldGpuContext, src_buf: &Buffer) -> Vec<[f32; 4]> {
        let num_cells = (self.params.width as usize) * (self.params.height as usize);
        let byte_size = (num_cells * 16) as u64;
        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Diagnostic Grid Readback"),
        });
        enc.copy_buffer_to_buffer(src_buf, 0, &self.field_staging, 0, byte_size);
        ctx.queue.submit(std::iter::once(enc.finish()));

        let slice = self.field_staging.slice(0..byte_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let out = {
            let view = slice.get_mapped_range().unwrap();
            let cells: &[[f32; 4]] = bytemuck::cast_slice(&view);
            cells.to_vec()
        };
        self.field_staging.unmap();
        out
    }
}
