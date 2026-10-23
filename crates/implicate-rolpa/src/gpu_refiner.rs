//! Hardware `wgpu` GPU compute pipeline for the 1-pass 4-block `CliffordNet-Lite` `@rolpa`
//! visual refiner (Track R3.3, Whitepaper §3.2 & §7.1).

use std::sync::OnceLock;
use std::time::Instant;

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
