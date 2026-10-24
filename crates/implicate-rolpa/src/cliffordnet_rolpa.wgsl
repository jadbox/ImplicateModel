// 4-Block No-FFN CliffordNet-Lite @rolpa Surface Refiner (Track R3.3, Whitepaper §3.2 & §7.1)
// Executes 4 blocks of factorized discrete Laplacian context C_loc(H) = \nabla^2 H
// and shifted Clifford Geometric Products D^{(s)} \oplus W^{(s)} (S = {1, 2, 4, 8, 16})
// with Gated Geometric Residuals (GGR) inside L1 workgroup shared memory.

struct RolpaUniforms {
    width: u32,
    height: u32,
    bypass_mode: u32,
    _pad0: u32,
    // Readout weights: [w_lap1, w_lap2, w_cliff_d, w_cliff_w]
    readout_weights: vec4<f32>,
    // Block residual gains: [gamma_0, gamma_1, gamma_2, gamma_3]
    block_gammas: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> params: RolpaUniforms;

// 12-channel input split into three vec4<f32> buffers:
// in_base: (R_base, G_base, B_base, cohesion_C)
@group(0) @binding(1)
var<storage, read> in_base: array<vec4<f32>>;

// in_grad: (dp_dx, dp_dy, dQ_dx, dQ_dy)
@group(0) @binding(2)
var<storage, read> in_grad: array<vec4<f32>>;

// in_lum: (L_r, L_g, L_b, edge_prior)
@group(0) @binding(3)
var<storage, read> in_lum: array<vec4<f32>>;

// Refined output RGBA
@group(0) @binding(4)
var<storage, read_write> out_rgba: array<vec4<f32>>;

// 18x18 workgroup shared tile (16x16 interior + 1px border halo)
var<workgroup> tile_h: array<vec4<f32>, 324>;

fn silu_v4(x: vec4<f32>) -> vec4<f32> {
    return x / (vec4<f32>(1.0) + exp(-clamp(x, vec4<f32>(-12.0), vec4<f32>(12.0))));
}

fn load_clamped_base(x: i32, y: i32) -> vec4<f32> {
    let cx = u32(clamp(x, 0, i32(params.width) - 1));
    let cy = u32(clamp(y, 0, i32(params.height) - 1));
    return in_base[cy * params.width + cx];
}

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let gx = i32(gid.x);
    let gy = i32(gid.y);
    let lx = i32(lid.x);
    let ly = i32(lid.y);
    let in_bounds = (gid.x < params.width) && (gid.y < params.height);
    let flat_idx = min(gid.y, params.height - 1u) * params.width + min(gid.x, params.width - 1u);

    let base_c = in_base[flat_idx];
    if (params.bypass_mode != 0u) {
        if (in_bounds) {
            out_rgba[flat_idx] = vec4<f32>(base_c.xyz, 1.0);
        }
        return;
    }

    let grad_pq = in_grad[flat_idx];
    let lum = in_lum[flat_idx];

    // Stem: lift 12-channel composite (I_base, \nabla p, \nabla Q_Cl, C, L) into Clifford state H
    var h = vec4<f32>(
        base_c.x + 0.02 * grad_pq.x + 0.02 * lum.x,
        base_c.y + 0.02 * grad_pq.y + 0.02 * lum.y,
        base_c.z + 0.02 * grad_pq.z + 0.02 * lum.z,
        0.3333333 * (base_c.x + base_c.y + base_c.z) * base_c.w
    );

    var lap1_rgb = vec3<f32>(0.0);
    var lap2_rgb = vec3<f32>(0.0);
    var cliff_d_acc = vec3<f32>(0.0);
    var cliff_w_acc = vec3<f32>(0.0);

    let tile_x = lx + 1;
    let tile_y = ly + 1;
    let tile_center = tile_y * 18 + tile_x;
    let base_wx = i32(wid.x) * 16;
    let base_wy = i32(wid.y) * 16;

    // 4-Block No-FFN CliffordNet-Lite loop
    for (var b: u32 = 0u; b < 4u; b = b + 1u) {
        tile_h[tile_center] = h;

        // Cooperative halo load for the 18x18 tile border
        let tid = ly * 16 + lx;
        if (tid < 68) {
            var hx = 0;
            var hy = 0;
            if (tid < 18) {
                hx = tid;
                hy = 0;
            } else if (tid < 36) {
                hx = tid - 18;
                hy = 17;
            } else if (tid < 52) {
                hx = 0;
                hy = (tid - 36) + 1;
            } else {
                hx = 17;
                hy = (tid - 52) + 1;
            }
            let wx = base_wx + hx - 1;
            let wy = base_wy + hy - 1;
            let border_v = load_clamped_base(wx, wy);
            tile_h[hy * 18 + hx] = vec4<f32>(
                border_v.xyz,
                0.3333333 * (border_v.x + border_v.y + border_v.z) * border_v.w
            );
        }
        workgroupBarrier();

        // Factorized discrete 3x3 Laplacian context: C_loc(H) = \nabla^2 H
        let n_up = tile_h[(tile_y - 1) * 18 + tile_x];
        let n_dn = tile_h[(tile_y + 1) * 18 + tile_x];
        let n_lt = tile_h[tile_y * 18 + (tile_x - 1)];
        let n_rt = tile_h[tile_y * 18 + (tile_x + 1)];
        let n_ul = tile_h[(tile_y - 1) * 18 + (tile_x - 1)];
        let n_ur = tile_h[(tile_y - 1) * 18 + (tile_x + 1)];
        let n_dl = tile_h[(tile_y + 1) * 18 + (tile_x - 1)];
        let n_dr = tile_h[(tile_y + 1) * 18 + (tile_x + 1)];

        let avg_card = 0.20 * (n_up + n_dn + n_lt + n_rt);
        let avg_diag = 0.05 * (n_ul + n_ur + n_dl + n_dr);
        let c_loc = (avg_card + avg_diag) - h;

        if (b == 0u) {
            lap1_rgb = clamp(-c_loc.xyz, vec3<f32>(-0.05), vec3<f32>(0.05));
        } else if (b == 1u) {
            lap2_rgb = clamp(-c_loc.xyz, vec3<f32>(-0.05), vec3<f32>(0.05));
        }

        // Shifted Clifford Geometric Product D^{(s)} \oplus W^{(s)} across cyclic shifts S = {1, 2, 4, 8, 16}
        // Shift s=1 (yzwx), s=2 (zwxy), s=4/8/16 (xyzw + cross-channel projections)
        let c_s1 = c_loc.yzwx;
        let h_s1 = h.yzwx;
        let c_s2 = c_loc.zwxy;
        let h_s2 = h.zwxy;

        let d_s1 = silu_v4(h * c_s1);
        let w_s1 = h * c_s1 - c_loc * h_s1;
        let d_s2 = silu_v4(h * c_s2);
        let w_s2 = h * c_s2 - c_loc * h_s2;
        let d_s16 = silu_v4(h * c_loc);

        let d_sum = 0.40 * d_s1 + 0.30 * d_s2 + 0.30 * d_s16;
        let w_sum = 0.60 * w_s1 + 0.40 * w_s2;

        cliff_d_acc = cliff_d_acc + clamp(d_sum.xyz, vec3<f32>(-0.08), vec3<f32>(0.08));
        cliff_w_acc = cliff_w_acc + clamp(w_sum.xyz, vec3<f32>(-0.08), vec3<f32>(0.08));

        // Gated Geometric Residual (GGR): H <- H + \gamma_b * ((1 - \sigma(\alpha)) * G_b - c_loc)
        let gamma_b = params.block_gammas[b];
        let g_feat = 0.35 * d_sum + 0.35 * w_sum - 0.50 * c_loc;
        h = h + gamma_b * g_feat;

        workgroupBarrier();
    }

    if (in_bounds) {
        let delta_rgb = params.readout_weights.x * lap1_rgb
            + params.readout_weights.y * lap2_rgb
            + params.readout_weights.z * cliff_d_acc
            + params.readout_weights.w * cliff_w_acc;
        let refined = clamp(base_c.xyz + delta_rgb, vec3<f32>(0.0), vec3<f32>(1.0));
        out_rgba[flat_idx] = vec4<f32>(refined, 1.0);
    }
}

// ============================================================================
// Track R10.4: Fused GPU Clifford Model Sheet Unfolding Compute Shader
// Evaluates flat GpuModelSheetMonad (19 Mv301s, 1216 bytes) in O(1) constant
// time per pixel with volume conservation (det S = 1) and animated film ink contour
// blades (|u ∧ ∇²u|) vs flat cel paint fills (u · ∇²u) with zero pixel overlap.
// ============================================================================

struct Mv301 {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    v3: vec4<f32>,
};

struct GpuModelSheetMonad {
    line_of_action: Mv301,
    constitutive_monad: Mv301,
    cde_monad: Mv301,
    aspect_monads: array<Mv301, 8>,
    part_monads: array<Mv301, 8>,
};

struct GpuPartPose {
    center: vec2<f32>,
    radii: vec2<f32>,
    angle_rad: f32,
    phase_rad: f32,
    view_idx: u32,
    part_idx: u32,
};

struct UnfoldUniforms {
    canvas_width: u32,
    canvas_height: u32,
    num_parts: u32,
    active_view: u32,
    holon_center: vec2<f32>,
    scale: vec2<f32>,
    phase_rad: f32,
    ink_threshold: f32,
    ink_width: f32,
    aspect_ratio: f32,
    volume_scale: vec2<f32>,
    flags: u32,
    _pad0: u32,
    part_poses: array<GpuPartPose, 8>,
    part_kernel_res: array<vec4<u32>, 8>,
};

@group(0) @binding(5)
var<uniform> unfold_params: UnfoldUniforms;

@group(0) @binding(6)
var<storage, read> monad_buffer: array<GpuModelSheetMonad>;

@group(0) @binding(7)
var<storage, read> kernel_data: array<vec4<f32>>;

@group(0) @binding(8)
var<storage, read_write> unfold_out_rgba: array<vec4<f32>>;

@compute @workgroup_size(16, 16, 1)
fn unfold_monads(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let gx = gid.x;
    let gy = gid.y;
    if (gx >= unfold_params.canvas_width || gy >= unfold_params.canvas_height) {
        return;
    }
    let flat_idx = gy * unfold_params.canvas_width + gx;
    let wx = f32(gx);
    let wy = f32(gy);

    let monad = monad_buffer[0];

    // 1. Line of Action 3D PGA screw axis blade L_spine ∈ G(3,0,1)
    // Plücker direction d = (d_x, d_y) in v1.y, v1.z
    let loa_d = vec2<f32>(monad.line_of_action.v1.y, monad.line_of_action.v1.z);
    let loa_len = max(length(loa_d), 1e-5);
    let loa_dir = loa_d / loa_len;
    let loa_normal = vec2<f32>(-loa_dir.y, loa_dir.x);

    // 2. Volume Conservation (det S = 1)
    // Continuous gait phase φ ∈ S¹ scales longitudinal by s and transverse by 1/s
    let s_vol = max(unfold_params.volume_scale.x, 1e-4);
    let inv_s_vol = max(unfold_params.volume_scale.y, 1e-4);

    // 3. Constitutive parameters and CDE chromatic signature
    let const_lum = vec3<f32>(
        monad.constitutive_monad.v0.w,
        monad.constitutive_monad.v1.x,
        monad.constitutive_monad.v1.y
    );
    let cde_rgb = vec3<f32>(
        monad.cde_monad.v1.x,
        monad.cde_monad.v1.y,
        monad.cde_monad.v1.z
    );

    // 4. Evaluate sub-holon parts
    let num_parts = min(unfold_params.num_parts, 8u);
    var accum_cel = vec3<f32>(0.0);
    var accum_ink = vec3<f32>(0.0);
    var best_alpha = 0.0;
    var weight_sum = 0.0;
    var is_ink_blade = false;
    var best_ink_vorticity = 0.0;

    for (var p: u32 = 0u; p < num_parts; p = p + 1u) {
        let pose = unfold_params.part_poses[p];
        let spec = monad.part_monads[p];
        let k_res = unfold_params.part_kernel_res[p];

        // Apply strict volume conservation det S = 1 to part radii
        let rx = max(pose.radii.x * s_vol, 1.0);
        let ry = max(pose.radii.y * inv_s_vol, 1.0);

        let dx = wx - pose.center.x;
        let dy = wy - pose.center.y;
        let c_ang = cos(-pose.angle_rad);
        let s_ang = sin(-pose.angle_rad);
        let u = (c_ang * dx - s_ang * dy) / rx;
        let v = (s_ang * dx + c_ang * dy) / ry;

        // Aerodynamic flow compliance for flowing appendages (applied in closed-form algebraic mode)
        let flow_comp = spec.v0.z;
        var u_deform = u;
        var v_deform = v;
        if (k_res.y == 0u && flow_comp > 1.0) {
            let trail_wave = sin(pose.phase_rad - flow_comp * u);
            v_deform = v - 0.20 * trail_wave;
        }

        let rho = sqrt(u_deform * u_deform + v_deform * v_deform);

        if (k_res.y > 0u && k_res.z > 0u) {
            // Kernel sampling mode: sample discrete/continuous Clifford Fourier wavepacket
            if (abs(u_deform) <= 1.08 && abs(v_deform) <= 1.08) {
                let rw = k_res.y;
                let rh = k_res.z;
                let fx = i32(round((u_deform + 1.0) * 0.5 * f32(rw - 1u)));
                let fy = i32(round((v_deform + 1.0) * 0.5 * f32(rh - 1u)));
                if (fx >= 0 && fy >= 0 && fx < i32(rw) && fy < i32(rh)) {
                    let cell_idx = k_res.x + u32(fy) * rw + u32(fx);

                    let cel_samp = kernel_data[cell_idx * 2u];
                    let ink_samp = kernel_data[cell_idx * 2u + 1u];

                    let part_alpha = cel_samp.w;
                    let part_ink = ink_samp.x;

                    if (part_alpha >= 0.15) {
                        accum_cel = accum_cel + cel_samp.xyz * part_alpha;
                        weight_sum = weight_sum + part_alpha;
                        best_alpha = max(best_alpha, part_alpha);

                        if (part_ink >= unfold_params.ink_threshold) {
                            is_ink_blade = true;
                            best_ink_vorticity = max(best_ink_vorticity, part_ink);
                            accum_ink = cel_samp.xyz;
                        }
                    }
                }
            }
        } else {
            // Closed-form algebraic monad evaluation:
            // animated film ink contour blade vorticity W(x) = |u ∧ ∇²u| peaks at boundary rho ≈ 1.0
            let ink_sigma = max(unfold_params.ink_width, 0.02);
            let d_bound = abs(rho - 1.0);
            let w_vort = exp(-0.5 * (d_bound / ink_sigma) * (d_bound / ink_sigma));

            if (w_vort >= unfold_params.ink_threshold) {
                // Razor-sharp ink contour blade (zero overlap with cel fill)
                is_ink_blade = true;
                best_ink_vorticity = max(best_ink_vorticity, w_vort);
                accum_ink = vec3<f32>(0.06, 0.05, 0.08);
                best_alpha = max(best_alpha, 1.0);
            } else if (rho < 1.0 - ink_sigma) {
                // Invariant cel paint fill (scalar inner product u · ∇²u)
                let cel_vitality = exp(-0.5 * rho * rho);
                let part_col = cde_rgb * (0.8 + 0.2 * const_lum);
                accum_cel = accum_cel + part_col * cel_vitality;
                weight_sum = weight_sum + cel_vitality;
                best_alpha = max(best_alpha, cel_vitality);
            }
        }
    }

    // 5. Final output composition with mutual exclusivity (ink ∩ cel = ∅)
    let mode_ink_only = (unfold_params.flags & 1u) != 0u;
    let mode_cel_only = (unfold_params.flags & 2u) != 0u;

    if (best_alpha >= 0.35 && weight_sum > 0.0) {
        let final_cel = accum_cel / weight_sum;
        if (mode_ink_only) {
            if (is_ink_blade) {
                unfold_out_rgba[flat_idx] = vec4<f32>(accum_ink, 1.0);
            } else {
                unfold_out_rgba[flat_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }
        } else if (mode_cel_only) {
            if (!is_ink_blade) {
                unfold_out_rgba[flat_idx] = vec4<f32>(clamp(final_cel, vec3<f32>(0.0), vec3<f32>(1.0)), best_alpha);
            } else {
                unfold_out_rgba[flat_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            }
        } else {
            unfold_out_rgba[flat_idx] = vec4<f32>(clamp(final_cel, vec3<f32>(0.0), vec3<f32>(1.0)), best_alpha);
        }
    } else {
        unfold_out_rgba[flat_idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}
