//! Hardware `wgpu` GPU compute pipeline for the 1-pass 4-block `CliffordNet-Lite` `@rolpa`
//! visual refiner (Track R3.3, Whitepaper §3.2 & §7.1).

use std::sync::OnceLock;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use implicate_lifter::{
    GpuModelSheetMonad, TurnaroundViewAspect, WavepacketPartKernel,
};
use wgpu::*;

use crate::cliffordnet::{RolpaInputGrid, RolpaUniforms};

#[derive(Clone)]
pub struct RolpaGpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

static SHARED_ROLPA_GPU: OnceLock<RolpaGpuContext> = OnceLock::new();

impl RolpaGpuContext {
    pub fn shared() -> &'static Self {
        SHARED_ROLPA_GPU.get_or_init(|| {
            pollster::block_on(Self::init_headless())
                .expect("implicate-rolpa requires a WGPU-capable GPU adapter")
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
                label: Some("implicate-rolpa CliffordNet-Lite GPU Device"),
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

/// Zero-readback VRAM pipeline executing the 4-block No-FFN `CliffordNet-Lite` `@rolpa` shader.
pub struct GpuCliffordNetRefiner {
    pub width: u32,
    pub height: u32,
    pub uniform_buffer: Buffer,
    pub buf_base: Buffer,
    pub buf_grad: Buffer,
    pub buf_lum: Buffer,
    pub buf_out: Buffer,
    pub buf_staging: Buffer,
    pub pipeline: ComputePipeline,
    pub bind_group: BindGroup,
}

impl GpuCliffordNetRefiner {
    pub fn new(ctx: &RolpaGpuContext, uniforms: RolpaUniforms) -> Self {
        let width = uniforms.width;
        let height = uniforms.height;
        let num_pixels = (width as u64) * (height as u64);
        let byte_size = num_pixels * 16; // vec4<f32> = 16 bytes per pixel

        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("cliffordnet_rolpa.wgsl"),
            source: ShaderSource::Wgsl(include_str!("cliffordnet_rolpa.wgsl").into()),
        });

        let uniform_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("RolpaUniforms"),
            size: std::mem::size_of::<RolpaUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let make_storage = |label: &str, rw: bool| -> Buffer {
            let mut usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
            if rw {
                usage |= BufferUsages::COPY_SRC;
            }
            ctx.device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: byte_size,
                usage,
                mapped_at_creation: false,
            })
        };

        let buf_base = make_storage("rolpa_in_base", false);
        let buf_grad = make_storage("rolpa_in_grad", false);
        let buf_lum = make_storage("rolpa_in_lum", false);
        let buf_out = make_storage("rolpa_out_rgba", true);

        let buf_staging = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("rolpa_staging_readback"),
            size: byte_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("CliffordNet-Lite BindGroupLayout"),
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
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 4,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("CliffordNet-Lite PipelineLayout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("CliffordNet-Lite ComputePipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("CliffordNet-Lite BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buf_base.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buf_grad.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buf_lum.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        Self {
            width,
            height,
            uniform_buffer,
            buf_base,
            buf_grad,
            buf_lum,
            buf_out,
            buf_staging,
            pipeline,
            bind_group,
        }
    }

    pub fn update_uniforms(&self, ctx: &RolpaGpuContext, uniforms: &RolpaUniforms) {
        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    pub fn upload_grid(&self, ctx: &RolpaGpuContext, grid: &RolpaInputGrid) {
        assert_eq!(grid.width as u32, self.width);
        assert_eq!(grid.height as u32, self.height);
        ctx.queue
            .write_buffer(&self.buf_base, 0, bytemuck::cast_slice(&grid.base_c));
        ctx.queue
            .write_buffer(&self.buf_grad, 0, bytemuck::cast_slice(&grid.grad_pq));
        ctx.queue
            .write_buffer(&self.buf_lum, 0, bytemuck::cast_slice(&grid.lum));
    }

    /// Encode a single 1-pass `CliffordNet-Lite` compute dispatch into `encoder`.
    pub fn encode_pass(&self, encoder: &mut CommandEncoder) {
        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("CliffordNet-Lite @rolpa 1-Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    /// Submit a single 1-pass `CliffordNet-Lite` compute dispatch on resident VRAM buffers.
    pub fn dispatch_single_pass(&self, ctx: &RolpaGpuContext) {
        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("CliffordNet-Lite @rolpa 1-Pass Submit"),
        });
        self.encode_pass(&mut enc);
        ctx.queue.submit(std::iter::once(enc.finish()));
    }

    /// Benchmark hardware GPU single-pass execution latency in milliseconds across `num_passes`
    /// synchronized dispatches on resident VRAM buffers.
    pub fn benchmark_forward_latency_ms(&self, ctx: &RolpaGpuContext, num_passes: usize) -> f64 {
        // Warmup passes
        for _ in 0..8 {
            let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Warmup"),
            });
            self.encode_pass(&mut enc);
            ctx.queue.submit(std::iter::once(enc.finish()));
        }
        let _ = ctx.device.poll(PollType::wait_indefinitely());

        let passes = num_passes.max(1);
        let t0 = Instant::now();
        for _ in 0..passes {
            let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Benchmark Pass"),
            });
            self.encode_pass(&mut enc);
            ctx.queue.submit(std::iter::once(enc.finish()));
        }
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        t0.elapsed().as_secs_f64() * 1000.0 / (passes as f64)
    }

    /// Execute a GPU forward pass and read back the refined `[f32; 3]` RGB image for quality verification.
    pub fn refine_and_readback(
        &self,
        ctx: &RolpaGpuContext,
        grid: &RolpaInputGrid,
    ) -> Vec<[f32; 3]> {
        self.upload_grid(ctx, grid);
        let byte_size = ((self.width as u64) * (self.height as u64)) * 16;

        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("CliffordNet-Lite Refine & Readback"),
        });
        self.encode_pass(&mut enc);
        enc.copy_buffer_to_buffer(&self.buf_out, 0, &self.buf_staging, 0, byte_size);
        ctx.queue.submit(std::iter::once(enc.finish()));

        let slice = self.buf_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let n = (self.width as usize) * (self.height as usize);
        let mut out = Vec::with_capacity(n);
        {
            let view = slice.get_mapped_range().unwrap();
            let rgba: &[[f32; 4]] = bytemuck::cast_slice(&view);
            for p in rgba.iter().take(n) {
                out.push([p[0], p[1], p[2]]);
            }
        }
        self.buf_staging.unmap();
        out
    }

    /// Execute a 1-pass GPU forward pass and read back the refined RGB8 buffer (`width * height * 3` bytes).
    pub fn refine_and_readback_u8(
        &self,
        ctx: &RolpaGpuContext,
        grid: &RolpaInputGrid,
        out_rgb: &mut [u8],
    ) {
        let n = (self.width as usize) * (self.height as usize);
        assert_eq!(out_rgb.len(), n * 3);
        self.upload_grid(ctx, grid);
        let byte_size = ((self.width as u64) * (self.height as u64)) * 16;

        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("CliffordNet-Lite Refine & Readback U8"),
        });
        self.encode_pass(&mut enc);
        enc.copy_buffer_to_buffer(&self.buf_out, 0, &self.buf_staging, 0, byte_size);
        ctx.queue.submit(std::iter::once(enc.finish()));

        let slice = self.buf_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        {
            let view = slice.get_mapped_range().unwrap();
            let rgba: &[[f32; 4]] = bytemuck::cast_slice(&view);
            for (idx, p) in rgba.iter().take(n).enumerate() {
                out_rgb[idx * 3] = (p[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgb[idx * 3 + 1] = (p[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgb[idx * 3 + 2] = (p[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        self.buf_staging.unmap();
    }
}

/// Dynamic runtime part pose passed to `GpuModelSheetUnfolder`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuPartPose {
    pub center: [f32; 2],
    pub radii: [f32; 2],
    pub angle_rad: f32,
    pub phase_rad: f32,
    pub view_idx: u32,
    pub part_idx: u32,
}

impl Default for GpuPartPose {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            radii: [16.0, 16.0],
            angle_rad: 0.0,
            phase_rad: 0.0,
            view_idx: 0,
            part_idx: 0,
        }
    }
}

/// Uniforms buffer (448 bytes) for `unfold_monads` single-pass compute shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct UnfoldUniforms {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub num_parts: u32,
    pub active_view: u32,
    pub holon_center: [f32; 2],
    pub scale: [f32; 2],
    pub phase_rad: f32,
    pub ink_threshold: f32,
    pub ink_width: f32,
    pub aspect_ratio: f32,
    pub volume_scale: [f32; 2],
    pub flags: u32,
    pub _pad0: u32,
    pub part_poses: [GpuPartPose; 8],
    pub part_kernel_res: [[u32; 4]; 8],
}

impl Default for UnfoldUniforms {
    fn default() -> Self {
        Self {
            canvas_width: 256,
            canvas_height: 240,
            num_parts: 1,
            active_view: 0,
            holon_center: [128.0, 120.0],
            scale: [1.0, 1.0],
            phase_rad: 0.0,
            ink_threshold: 0.12,
            ink_width: 0.08,
            aspect_ratio: 1.0,
            volume_scale: [1.0, 1.0],
            flags: 0,
            _pad0: 0,
            part_poses: [GpuPartPose::default(); 8],
            part_kernel_res: [[0; 4]; 8],
        }
    }
}

/// Zero-readback hardware GPU compute pipeline executing the fused `unfold_monads` shader
/// evaluating flat 1216-byte `GpuModelSheetMonad` monads with volume conservation (det S = 1)
/// and animated film ink contour blades (|u ∧ ∇²u|) vs cel paint fills (u · ∇²u).
pub struct GpuModelSheetUnfolder {
    pub width: u32,
    pub height: u32,
    pub uniform_buffer: Buffer,
    pub monad_buffer: Buffer,
    pub kernel_buffer: Buffer,
    pub out_buffer: Buffer,
    pub staging_buffer: Buffer,
    pub pipeline: ComputePipeline,
    pub bind_group: BindGroup,
}

impl GpuModelSheetUnfolder {
    pub fn new(ctx: &RolpaGpuContext, width: u32, height: u32) -> Self {
        let num_pixels = (width as u64) * (height as u64);
        let byte_size = num_pixels * 16; // vec4<f32> per pixel

        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("cliffordnet_rolpa_unfold.wgsl"),
            source: ShaderSource::Wgsl(include_str!("cliffordnet_rolpa.wgsl").into()),
        });

        let uniform_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("UnfoldUniforms"),
            size: std::mem::size_of::<UnfoldUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let monad_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("GpuModelSheetMonad"),
            size: std::mem::size_of::<GpuModelSheetMonad>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kernel_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("UnfoldKernelData"),
            size: 16 * 1024 * 1024, // 16 MB kernel storage capacity
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("UnfoldOutRGBA"),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: Some("UnfoldStagingReadback"),
            size: byte_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("GpuModelSheetUnfolder BindGroupLayout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 5,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 6,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 7,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 8,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("GpuModelSheetUnfolder PipelineLayout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("GpuModelSheetUnfolder ComputePipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("unfold_monads"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("GpuModelSheetUnfolder BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 5,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: monad_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: kernel_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: out_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            width,
            height,
            uniform_buffer,
            monad_buffer,
            kernel_buffer,
            out_buffer,
            staging_buffer,
            pipeline,
            bind_group,
        }
    }

    pub fn upload_monad(&self, ctx: &RolpaGpuContext, monad: &GpuModelSheetMonad) {
        ctx.queue
            .write_buffer(&self.monad_buffer, 0, bytemuck::bytes_of(monad));
    }

    pub fn upload_uniforms(&self, ctx: &RolpaGpuContext, uniforms: &UnfoldUniforms) {
        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    pub fn upload_kernel_data(&self, ctx: &RolpaGpuContext, kernel_data: &[[f32; 4]]) {
        if !kernel_data.is_empty() {
            ctx.queue.write_buffer(
                &self.kernel_buffer,
                0,
                bytemuck::cast_slice(kernel_data),
            );
        }
    }

    pub fn encode_pass(&self, encoder: &mut CommandEncoder) {
        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("GpuModelSheetUnfolder 1-Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    pub fn dispatch_single_pass(&self, ctx: &RolpaGpuContext) {
        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("GpuModelSheetUnfolder 1-Pass Submit"),
        });
        self.encode_pass(&mut enc);
        ctx.queue.submit(std::iter::once(enc.finish()));
    }

    /// Benchmark hardware GPU single-pass unfold latency in milliseconds across `num_passes`
    /// synchronized dispatches on resident VRAM buffers.
    pub fn benchmark_unfold_latency_ms(&self, ctx: &RolpaGpuContext, num_passes: usize) -> f64 {
        // Warmup passes
        for _ in 0..8 {
            let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Unfold Warmup"),
            });
            self.encode_pass(&mut enc);
            ctx.queue.submit(std::iter::once(enc.finish()));
        }
        let _ = ctx.device.poll(PollType::wait_indefinitely());

        let passes = num_passes.max(1);
        let t0 = Instant::now();
        for _ in 0..passes {
            let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Unfold Benchmark Pass"),
            });
            self.encode_pass(&mut enc);
            ctx.queue.submit(std::iter::once(enc.finish()));
        }
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        t0.elapsed().as_secs_f64() * 1000.0 / (passes as f64)
    }

    /// Execute a 1-pass GPU unfold and read back the refined RGB8 and mask buffers.
    pub fn unfold_and_readback_u8(
        &self,
        ctx: &RolpaGpuContext,
        monad: &GpuModelSheetMonad,
        uniforms: &UnfoldUniforms,
        kernel_data: Option<&[[f32; 4]]>,
        out_rgb: &mut [u8],
        out_mask: &mut [bool],
    ) {
        let n = (self.width as usize) * (self.height as usize);
        assert_eq!(out_rgb.len(), n * 3);
        assert_eq!(out_mask.len(), n);

        self.upload_monad(ctx, monad);
        self.upload_uniforms(ctx, uniforms);
        if let Some(kd) = kernel_data {
            self.upload_kernel_data(ctx, kd);
        }

        let byte_size = ((self.width as u64) * (self.height as u64)) * 16;
        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("GpuModelSheetUnfolder Readback U8"),
        });
        self.encode_pass(&mut enc);
        enc.copy_buffer_to_buffer(&self.out_buffer, 0, &self.staging_buffer, 0, byte_size);
        ctx.queue.submit(std::iter::once(enc.finish()));

        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        {
            let view = slice.get_mapped_range().unwrap();
            let rgba: &[[f32; 4]] = bytemuck::cast_slice(&view);
            for (idx, p) in rgba.iter().take(n).enumerate() {
                let alpha = p[3];
                out_mask[idx] = alpha >= 0.35;
                out_rgb[idx * 3] = (p[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgb[idx * 3 + 1] = (p[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                out_rgb[idx * 3 + 2] = (p[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        self.staging_buffer.unmap();
    }

    /// Execute a 1-pass GPU unfold and read back the RGBA float buffer.
    pub fn unfold_and_readback_rgba(
        &self,
        ctx: &RolpaGpuContext,
        monad: &GpuModelSheetMonad,
        uniforms: &UnfoldUniforms,
        kernel_data: Option<&[[f32; 4]]>,
    ) -> Vec<[f32; 4]> {
        let n = (self.width as usize) * (self.height as usize);
        self.upload_monad(ctx, monad);
        self.upload_uniforms(ctx, uniforms);
        if let Some(kd) = kernel_data {
            self.upload_kernel_data(ctx, kd);
        }

        let byte_size = ((self.width as u64) * (self.height as u64)) * 16;
        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("GpuModelSheetUnfolder Readback RGBA"),
        });
        self.encode_pass(&mut enc);
        enc.copy_buffer_to_buffer(&self.out_buffer, 0, &self.staging_buffer, 0, byte_size);
        ctx.queue.submit(std::iter::once(enc.finish()));

        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        let _ = ctx.device.poll(PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let mut out = Vec::with_capacity(n);
        {
            let view = slice.get_mapped_range().unwrap();
            let rgba: &[[f32; 4]] = bytemuck::cast_slice(&view);
            out.extend_from_slice(&rgba[..n]);
        }
        self.staging_buffer.unmap();
        out
    }

    /// Pack a `TurnaroundViewAspect`'s part kernels into flat GPU storage buffer `[[f32; 4]]`
    /// and return `(kernel_data, part_kernel_res)`.
    pub fn pack_model_sheet_view_kernels(
        view: &TurnaroundViewAspect,
    ) -> (Vec<[f32; 4]>, [[u32; 4]; 8]) {
        let mut kernel_data = Vec::new();
        let mut part_kernel_res = [[0u32; 4]; 8];

        for (p, k) in view.part_kernels.iter().take(8).enumerate() {
            let rw = k.canonical_res[0];
            let rh = k.canonical_res[1];
            if rw == 0 || rh == 0 {
                continue;
            }
            let offset = (kernel_data.len() / 2) as u32;
            part_kernel_res[p] = [offset, rw as u32, rh as u32, 0];

            let n = rw * rh;
            for i in 0..n {
                let (r, g, b, a) = if i * 4 + 3 < k.cel_fill_rgb.len() {
                    (
                        k.cel_fill_rgb[i * 4] as f32 / 255.0,
                        k.cel_fill_rgb[i * 4 + 1] as f32 / 255.0,
                        k.cel_fill_rgb[i * 4 + 2] as f32 / 255.0,
                        k.cel_fill_rgb[i * 4 + 3] as f32 / 255.0,
                    )
                } else if i * 3 + 2 < k.cel_fill_rgb.len() {
                    (
                        k.cel_fill_rgb[i * 3] as f32 / 255.0,
                        k.cel_fill_rgb[i * 3 + 1] as f32 / 255.0,
                        k.cel_fill_rgb[i * 3 + 2] as f32 / 255.0,
                        1.0,
                    )
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };
                let ink = k.ink_contour_blade.get(i).copied().unwrap_or(0.0);
                let cft = k.cft_deformation_coeffs.get(i).copied().unwrap_or([0.0; 5]);
                kernel_data.push([r, g, b, a]);
                kernel_data.push([ink, cft[1], cft[2], cft[3]]);
            }
        }

        (kernel_data, part_kernel_res)
    }

    /// Pack `WavepacketPartKernel`s into flat GPU storage buffer `[[f32; 4]]`
    /// evaluating continuous 5-mode Clifford Fourier expansion at `phase_rad`
    /// and return `(kernel_data, part_kernel_res)`.
    pub fn pack_wavepacket_kernels_for_phase(
        kernels: &[WavepacketPartKernel],
        phase_rad: f64,
    ) -> (Vec<[f32; 4]>, [[u32; 4]; 8]) {
        let mut kernel_data = Vec::new();
        let mut part_kernel_res = [[0u32; 4]; 8];
        let (s1, c1) = phase_rad.sin_cos();
        let (s2, c2) = (2.0 * phase_rad).sin_cos();
        let b = [1.0f32, c1 as f32, s1 as f32, c2 as f32, s2 as f32];

        for (p, k) in kernels.iter().take(8).enumerate() {
            let rw = k.res_w;
            let rh = k.res_h;
            if rw == 0 || rh == 0 {
                continue;
            }
            let offset = (kernel_data.len() / 2) as u32;
            part_kernel_res[p] = [offset, rw as u32, rh as u32, 0];

            let n = rw * rh;
            for i in 0..n {
                let (r, g, b_col, a) = if !k.cft_rgba_coeffs.is_empty() && (i * 4 + 3) < k.cft_rgba_coeffs.len() {
                    let eval_ch = |ch: usize| -> f32 {
                        let w = &k.cft_rgba_coeffs[i * 4 + ch];
                        w[0] * b[0] + w[1] * b[1] + w[2] * b[2] + w[3] * b[3] + w[4] * b[4]
                    };
                    let env = k.gaussian_envelope.get(i).copied().unwrap_or(1.0);
                    let alpha = (eval_ch(3) / 255.0).clamp(0.0, 1.0) * env;
                    let r = (eval_ch(0) / 255.0).clamp(0.0, 1.0);
                    let g = (eval_ch(1) / 255.0).clamp(0.0, 1.0);
                    let bl = (eval_ch(2) / 255.0).clamp(0.0, 1.0);
                    (r, g, bl, alpha)
                } else if i * 4 + 3 < k.dc_rgba.len() {
                    (
                        k.dc_rgba[i * 4] as f32 / 255.0,
                        k.dc_rgba[i * 4 + 1] as f32 / 255.0,
                        k.dc_rgba[i * 4 + 2] as f32 / 255.0,
                        k.dc_rgba[i * 4 + 3] as f32 / 255.0,
                    )
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };
                let ink = k.ink_contour_blade.get(i).copied().unwrap_or(0.0);
                kernel_data.push([r, g, b_col, a]);
                kernel_data.push([ink, 0.0, 0.0, 0.0]);
            }
        }

        (kernel_data, part_kernel_res)
    }

    /// Pack `WavepacketPartKernel`s into flat GPU storage buffer `[[f32; 4]]`
    /// and return `(kernel_data, part_kernel_res)`.
    pub fn pack_wavepacket_kernels(
        kernels: &[WavepacketPartKernel],
    ) -> (Vec<[f32; 4]>, [[u32; 4]; 8]) {
        Self::pack_wavepacket_kernels_for_phase(kernels, 0.0)
    }
}
