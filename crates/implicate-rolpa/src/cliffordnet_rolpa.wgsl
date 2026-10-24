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
