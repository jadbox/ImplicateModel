// Unified Staggered Yee FDTD Wave-Matter Lattice + Chromatic Luminescence + Zero-Readback Holon Reduction
// (Track R4.1, Whitepaper §6.2 & §6.3)

struct LatticeParams {
    width: u32,
    height: u32,
    num_sources: u32,
    num_queries: u32,
    dx: f32,
    dy: f32,
    dt: f32,
    inv_dx: f32,
    inv_dy: f32,
    beta: f32,
    alpha: f32,
    threshold_decay: f32,
    kerr_lensing: f32,
    bohmian_gain: f32,
    outer_periodic: u32,
    _pad0: u32,
};

// Active Holon wake & luminescence emitter (32 bytes)
struct GpuHolonSource {
    // (x, y, vx, vy) in grid coordinates
    kinematics: vec4<f32>,
    // (L_r, L_g, L_b, radius)
    emission: vec4<f32>,
};

// Holon / Binaural Ear Probe query (16 bytes)
struct GpuHolonQuery {
    x: f32,
    y: f32,
    radius: f32,
    yield_pressure: f32,
};

// Zero-readback reduction record per holon/probe (32 bytes)
struct GpuHolonReduction {
    // (fx, fy, tau_z, max_pressure)
    forces: vec4<f32>,
    // (flow_vx, flow_vy, grad_lum_x, grad_lum_y)
    field_env: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: LatticeParams;
// state_a / state_b: vec4<f32> = (pressure_p, vx_east, vy_north, bohmian_Q)
@group(0) @binding(1) var<storage, read> state_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> state_out: array<vec4<f32>>;
// material: vec4<f32> = (rho, K, gamma, cohesion_C)
@group(0) @binding(3) var<storage, read_write> material: array<vec4<f32>>;
// lum_in / lum_out: vec4<f32> = (L_r, L_g, L_b, p_yield)
@group(0) @binding(4) var<storage, read> lum_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> lum_out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> sources: array<GpuHolonSource>;
@group(0) @binding(7) var<storage, read> queries: array<GpuHolonQuery>;
@group(0) @binding(8) var<storage, read_write> reductions: array<GpuHolonReduction>;

fn wrap_or_clamp(c: i32, limit: u32, periodic: u32) -> u32 {
    let lim = i32(limit);
    if (periodic != 0u) {
        return u32(((c % lim) + lim) % lim);
    }
    return u32(clamp(c, 0, lim - 1));
}

fn cell_idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

// Pass 1: Staggered Yee Velocity Update + Bohm-Hiley Q = -(nabla^2 R)/R + Moving Holon Doppler Wake Injection
@compute @workgroup_size(16, 16, 1)
fn step_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let w = params.width;
    let h = params.height;
    let idx = cell_idx(x, y);
    let cur_s = state_in[idx];
    let cur_m = material[idx];

    let xe = wrap_or_clamp(i32(x) + 1, w, params.outer_periodic);
    let xw = wrap_or_clamp(i32(x) - 1, w, params.outer_periodic);
    let yn = wrap_or_clamp(i32(y) + 1, h, params.outer_periodic);
    let ys = wrap_or_clamp(i32(y) - 1, h, params.outer_periodic);

    let s_e = state_in[cell_idx(xe, y)];
    let s_w = state_in[cell_idx(xw, y)];
    let s_n = state_in[cell_idx(x, yn)];
    let s_s = state_in[cell_idx(x, ys)];

    let m_e = material[cell_idx(xe, y)];
    let m_n = material[cell_idx(x, yn)];

    let p_c = cur_s.x;
    var vx_new = cur_s.y;
    var vy_new = cur_s.z;

    // East face velocity update
    if (x + 1u < w || params.outer_periodic != 0u) {
        let grad_px = (s_e.x - p_c) * params.inv_dx;
        let rho_x = max(0.5 * (cur_m.x + m_e.x), 0.05);
        let gamma_x = 0.5 * (cur_m.z + m_e.z);
        let decay_x = exp(-gamma_x * params.dt);
        vx_new = decay_x * cur_s.y - (grad_px / rho_x) * params.dt;
    } else {
        vx_new = 0.0;
    }

    // North face velocity update
    if (y + 1u < h || params.outer_periodic != 0u) {
        let grad_py = (s_n.x - p_c) * params.inv_dy;
        let rho_y = max(0.5 * (cur_m.x + m_n.x), 0.05);
        let gamma_y = 0.5 * (cur_m.z + m_n.z);
        let decay_y = exp(-gamma_y * params.dt);
        vy_new = decay_y * cur_s.z - (grad_py / rho_y) * params.dt;
    } else {
        vy_new = 0.0;
    }

    // Inject moving Holon Doppler dipole wake into velocity & pressure
    var p_wake = 0.0;
    let fx = f32(x);
    let fy = f32(y);
    for (var s_i: u32 = 0u; s_i < params.num_sources; s_i = s_i + 1u) {
        let src = sources[s_i];
        let dx = fx - src.kinematics.x;
        let dy = fy - src.kinematics.y;
        let r_src = max(src.emission.w, 1.5);
        let two_sig2 = 2.0 * r_src * r_src;
        let dist2 = dx * dx + dy * dy;
        if (dist2 <= 6.0 * two_sig2) {
            let env = exp(-dist2 / two_sig2);
            let svx = src.kinematics.z;
            let svy = src.kinematics.w;
            // Dipole compression ahead of motion (+v . r) and rarefaction behind (-v . r)
            let doppler_proj = (dx * svx + dy * svy) / r_src;
            p_wake = p_wake + 0.12 * doppler_proj * env * params.dt;
            vx_new = vx_new + 0.22 * svx * env * params.dt;
            vy_new = vy_new + 0.22 * svy * env * params.dt;
        }
    }

    // Scale-invariant Bohm-Hiley Quantum Potential Q = -(nabla^2 R) / R
    let eps_sq = 0.0064;
    let r_c = sqrt(p_c * p_c + eps_sq);
    let r_e = sqrt(s_e.x * s_e.x + eps_sq);
    let r_w = sqrt(s_w.x * s_w.x + eps_sq);
    let r_n = sqrt(s_n.x * s_n.x + eps_sq);
    let r_s = sqrt(s_s.x * s_s.x + eps_sq);
    let lap_r = (r_e - 2.0 * r_c + r_w) * (params.inv_dx * params.inv_dx)
              + (r_n - 2.0 * r_c + r_s) * (params.inv_dy * params.inv_dy);
    let q_bohm = clamp(-lap_r / r_c, -25.0, 25.0);

    state_out[idx] = vec4<f32>(p_c + p_wake, vx_new, vy_new, q_bohm);
}

// Pass 2: Staggered Yee Pressure Update + Kerr Lensing + Rational de Broglie Soliton + Griffith Cohesion Erosion + Chromatic Luminescence
@compute @workgroup_size(16, 16, 1)
fn step_pressure_and_luminescence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }

    let w = params.width;
    let h = params.height;
    let idx = cell_idx(x, y);
    let cur_s = state_out[idx];
    var cur_m = material[idx];
    let cur_l = lum_in[idx];

    let xe = wrap_or_clamp(i32(x) + 1, w, params.outer_periodic);
    let xw = wrap_or_clamp(i32(x) - 1, w, params.outer_periodic);
    let yn = wrap_or_clamp(i32(y) + 1, h, params.outer_periodic);
    let ys = wrap_or_clamp(i32(y) - 1, h, params.outer_periodic);

    let vx_e = cur_s.y;
    let vy_n = cur_s.z;
    var vx_w = 0.0;
    if (x > 0u || params.outer_periodic != 0u) {
        vx_w = state_out[cell_idx(xw, y)].y;
    }
    var vy_s = 0.0;
    if (y > 0u || params.outer_periodic != 0u) {
        vy_s = state_out[cell_idx(x, ys)].z;
    }

    let div_v = (vx_e - vx_w) * params.inv_dx + (vy_n - vy_s) * params.inv_dy;
    let k_mod = cur_m.y;
    let gamma = cur_m.z;
    let decay_p = exp(-gamma * params.dt);

    let p_old = cur_s.x;
    var k_eff = k_mod;
    if (params.kerr_lensing > 0.0) {
        let p2 = p_old * p_old;
        let lens = (params.kerr_lensing * p2) / (1.0 + params.alpha * p2);
        k_eff = k_mod * max(0.35, 1.0 - lens);
    }

    let p_lin = decay_p * p_old - (k_eff * params.dt) * div_v;
    var p_new = p_lin;
    if (params.beta > 0.0) {
        let p2 = p_lin * p_lin;
        let f_nl = (params.beta * p_lin) / (1.0 + params.alpha * p2) - params.threshold_decay * p_lin;
        p_new = p_lin + f_nl * params.dt;
    }

    // Griffith cohesion erosion when overpressure exceeds yield threshold
    let p_abs = abs(p_new);
    let p_yield = max(cur_l.w, 0.85);
    if (p_abs > p_yield && cur_m.w > 0.05) {
        cur_m.w = max(0.05, cur_m.w - 0.15 * (p_abs - p_yield) * params.dt);
        material[idx] = cur_m;
    }

    // Screened reaction-diffusion for 3-channel chromatic luminescence L = (L_r, L_g, L_b)
    let l_e = lum_in[cell_idx(xe, y)].xyz;
    let l_w = lum_in[cell_idx(xw, y)].xyz;
    let l_n = lum_in[cell_idx(x, yn)].xyz;
    let l_s = lum_in[cell_idx(x, ys)].xyz;
    let l_c = cur_l.xyz;

    let lap_l = (l_e - 2.0 * l_c + l_w) * (params.inv_dx * params.inv_dx)
              + (l_n - 2.0 * l_c + l_s) * (params.inv_dy * params.inv_dy);
    let c_speed = sqrt(max(cur_m.y / max(cur_m.x, 0.05), 0.1));
    let diff_l = 0.22 * c_speed;

    var emit_rgb = vec3<f32>(0.0);
    if (p_abs > 0.35) {
        let w_amp = (p_abs - 0.35) * 0.35;
        emit_rgb = emit_rgb + vec3<f32>(w_amp * 0.8, w_amp * 0.9, w_amp);
    }

    let fx = f32(x);
    let fy = f32(y);
    for (var s_i: u32 = 0u; s_i < params.num_sources; s_i = s_i + 1u) {
        let src = sources[s_i];
        let l_src = src.emission.xyz;
        if (l_src.x + l_src.y + l_src.z > 0.001) {
            let dx = fx - src.kinematics.x;
            let dy = fy - src.kinematics.y;
            let r_src = max(src.emission.w, 1.5);
            let two_sig2 = 2.5 * r_src * r_src;
            let dist2 = dx * dx + dy * dy;
            if (dist2 <= 8.0 * two_sig2) {
                emit_rgb = emit_rgb + l_src * exp(-dist2 / two_sig2);
            }
        }
    }

    let decay_l = exp(-1.65 * params.dt);
    let l_next = clamp(
        decay_l * (l_c + diff_l * lap_l * params.dt) + emit_rgb * params.dt,
        vec3<f32>(0.0),
        vec3<f32>(6.0)
    );

    state_out[idx] = vec4<f32>(p_new, vx_e, vy_n, cur_s.w);
    lum_out[idx] = vec4<f32>(l_next, p_yield);
}

// Pass 3: Zero-Readback GPU Force, Bohmian Guidance, Fluid Velocity & Luminescence Gradient Reduction
@compute @workgroup_size(32, 1, 1)
fn reduce_holons(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_idx = gid.x;
    if (q_idx >= params.num_queries) {
        return;
    }

    let q = queries[q_idx];
    let w = params.width;
    let h = params.height;

    // Point probe (e.g. binaural camera ear pickup): return (vx, vy, Q, p) and (0, 0, L_r, L_b)
    if (q.radius <= 0.05) {
        let px = wrap_or_clamp(i32(round(q.x)), w, params.outer_periodic);
        let py = wrap_or_clamp(i32(round(q.y)), h, params.outer_periodic);
        let s = state_out[cell_idx(px, py)];
        let l = lum_out[cell_idx(px, py)];
        reductions[q_idx] = GpuHolonReduction(
            vec4<f32>(s.y, s.z, s.w, s.x),
            vec4<f32>(s.y, s.z, l.x, l.z)
        );
        return;
    }

    let r_cells = min(i32(ceil(q.radius)) + 1, 12);
    let r_sq_max = (q.radius + 1.0) * (q.radius + 1.0);
    let cx = i32(round(q.x));
    let cy = i32(round(q.y));

    var fx_total = 0.0;
    var fy_total = 0.0;
    var tau_total = 0.0;
    var max_p = 0.0;
    var flow_vx = 0.0;
    var flow_vy = 0.0;
    var grad_lx = 0.0;
    var grad_ly = 0.0;
    var count = 0.0;

    for (var oy = -r_cells; oy <= r_cells; oy = oy + 1) {
        for (var ox = -r_cells; ox <= r_cells; ox = ox + 1) {
            let rx = f32(cx + ox) - q.x;
            let ry = f32(cy + oy) - q.y;
            if (rx * rx + ry * ry > r_sq_max) {
                continue;
            }

            let gx = wrap_or_clamp(cx + ox, w, params.outer_periodic);
            let gy = wrap_or_clamp(cy + oy, h, params.outer_periodic);
            let xe = wrap_or_clamp(i32(gx) + 1, w, params.outer_periodic);
            let xw = wrap_or_clamp(i32(gx) - 1, w, params.outer_periodic);
            let yn = wrap_or_clamp(i32(gy) + 1, h, params.outer_periodic);
            let ys = wrap_or_clamp(i32(gy) - 1, h, params.outer_periodic);

            let s_c = state_out[cell_idx(gx, gy)];
            let s_e = state_out[cell_idx(xe, gy)];
            let s_w = state_out[cell_idx(xw, gy)];
            let s_n = state_out[cell_idx(gx, yn)];
            let s_s = state_out[cell_idx(gx, ys)];

            let l_e = lum_out[cell_idx(xe, gy)].xyz;
            let l_w = lum_out[cell_idx(xw, gy)].xyz;
            let l_n = lum_out[cell_idx(gx, yn)].xyz;
            let l_s = lum_out[cell_idx(gx, ys)].xyz;

            let p_abs = abs(s_c.x);
            if (p_abs > max_p) {
                max_p = p_abs;
            }

            let grad_px = (s_e.x - s_w.x) * 0.5 * params.inv_dx;
            let grad_py = (s_n.x - s_s.x) * 0.5 * params.inv_dy;
            let q_fx = -(s_e.w - s_w.w) * 0.5 * params.inv_dx;
            let q_fy = -(s_n.w - s_s.w) * 0.5 * params.inv_dy;

            let dfx = -grad_px * 0.18 + q_fx * params.bohmian_gain;
            let dfy = -grad_py * 0.18 + q_fy * params.bohmian_gain;

            fx_total = fx_total + dfx;
            fy_total = fy_total + dfy;
            tau_total = tau_total + (rx * dfy - ry * dfx);
            flow_vx = flow_vx + s_c.y;
            flow_vy = flow_vy + s_c.z;

            let lum_mag_e = l_e.x + l_e.y + l_e.z;
            let lum_mag_w = l_w.x + l_w.y + l_w.z;
            let lum_mag_n = l_n.x + l_n.y + l_n.z;
            let lum_mag_s = l_s.x + l_s.y + l_s.z;
            grad_lx = grad_lx + (lum_mag_e - lum_mag_w) * 0.5 * params.inv_dx;
            grad_ly = grad_ly + (lum_mag_n - lum_mag_s) * 0.5 * params.inv_dy;
            count = count + 1.0;
        }
    }

    let inv_c = 1.0 / max(count, 1.0);
    reductions[q_idx] = GpuHolonReduction(
        vec4<f32>(fx_total, fy_total, tau_total, max_p),
        vec4<f32>(flow_vx * inv_c, flow_vy * inv_c, grad_lx * inv_c, grad_ly * inv_c)
    );
}
