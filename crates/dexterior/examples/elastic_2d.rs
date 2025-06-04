//! A complex example making use of `dexterior`'s entire feature set.
//! An elastic wave pulse propagates through a layered isotropic material
//! with material parameters that are locally constant in each layer.
//!
//! This is a prototype of a phononic heat transfer simulation
//! which is planned to eventually be published as a paper.
//! (TODO: once that project lives in a different place,
//! we should probably remove some of the less-relevant code
//! to make this a work better as an example.)
//! More details will be found there once published;
//! for now, a brief description:
//!
//! The elastic wave equation
//! ϼ∂ₜₜu - (λ + 2μ)∇(∇·u) + μ∇²u = 0
//! is split into two scalar wave equations (pressure and shear waves),
//! which are coupled only at material boundaries.
//! These equations are expressed as the first-order system
//! ∂ₜp - (λ + 2μ)∇·v = 0
//! ∂ₜw + μ∇×v = 0
//! ϼ∂ₜv - ∇p - ∇×w = 0
//! where
//! v = ∂ₜu = ∇Φ + ∇×Ψ,
//! p = ϼ∂ₜɸ,
//! w = ϼ∂ₜΨ.
//! The velocity v is stored on primal mesh edges,
//! where it is further split into tangential and normal components
//! in order to facilitate absorbing boundary conditions,
//! which are implemented as a first-order Engqvist-Majda condition
//! similar to the acoustic scatterer example.

use std::f64::consts::{PI, TAU};

use dexterior as dex;
use dexterior_visuals as dv;

type Pressure = dex::Cochain<0, dex::Dual>;
type Flux = dex::Cochain<1, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;
type Shear = dex::Cochain<0, dex::Dual>;

#[derive(Clone, Debug)]
struct State {
    t: f64,
    p: Pressure,
    q: Flux,
    w: Shear,
    v: Velocity,
    // false to draw the shear potential, true to draw pressure
    // (these parameters stored in state for keyboard controls)
    draw_pressure: bool,
    draw_arrows: bool,
}

impl dv::AnimationState for State {
    fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
        Self {
            t: old.t + t * (new.t - old.t),
            p: old.p.lerp(&new.p, t),
            q: old.q.lerp(&new.q, t),
            w: old.w.lerp(&new.w, t),
            v: old.v.lerp(&new.v, t),
            draw_pressure: old.draw_pressure,
            draw_arrows: old.draw_arrows,
        }
    }
}

struct Ops {
    p_step: dex::Op<Flux, Pressure>,
    q_step: dex::Op<Pressure, Flux>,
    // operators that apply pressure interpolated onto primal vertices
    // into the shear wave and shear into the pressure wave.
    // these only have an effect at material boundaries
    q_step_interp: dex::Op<Shear, Flux>,
    w_step: dex::Op<Velocity, Shear>,
    v_step: dex::Op<Shear, Velocity>,
    v_step_interp: dex::Op<Pressure, Velocity>,
}

struct MaterialArea {
    edges: dex::Subset<1, dex::Primal>,
    tris: dex::Subset<2, dex::Primal>,
    boundary: dex::Subset<1, dex::Primal>,
    // lamè coefficients and other parameters of the material in this area
    // (stiffness = λ + 2μ)
    mu: f64,
    density: f64,
    stiffness: f64,
    p_wave_speed: f64,
    s_wave_speed: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let msh_bytes = include_bytes!("./meshes/2d_layered.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    // spatially varying material parameters from mesh physical groups

    let mut layers: Vec<MaterialArea> = Vec::new();
    // loop until no more layers found
    // instead of hardcoding layer count
    // so that we can easily change this via gmsh parameters
    let mut layer = 1;
    loop {
        let group_id = format!("{layer}");
        let (Some(edges), Some(tris)) = (
            mesh.get_subset::<1>(&group_id),
            mesh.get_subset::<2>(&group_id),
        ) else {
            break;
        };

        let boundary = tris.manifold_boundary(&mesh);

        let lambda = 1.;
        let mu = 1.;
        let density = layer as f64;
        let stiffness = lambda + 2. * mu;

        let p_wave_speed = f64::sqrt(stiffness / density);
        let s_wave_speed = f64::sqrt(mu / density);

        layers.push(MaterialArea {
            edges,
            tris,
            boundary,
            mu,
            density,
            stiffness,
            p_wave_speed,
            s_wave_speed,
        });
        layer += 1;
    }

    let boundary_edges = mesh.boundary::<1>();
    let bottom_edges = mesh.get_subset::<1>("990").expect("Subset not found");

    let mut layer_boundary_edges = layers[0].boundary.clone();
    for layer in layers.iter().skip(1) {
        layer_boundary_edges = layer_boundary_edges.union(&layer.boundary);
    }
    layer_boundary_edges = layer_boundary_edges.difference(&boundary_edges);

    // other parameters

    let dt = 1. / 60.;
    // whether to send p and/or s waves
    // (useful to turn one off to see refraction/reflection patterns of the other)
    let send_p_pulse = true;
    let send_s_pulse = true;
    let color_map_range = 2.;

    // angle of wave propagation between 0 and 90 degrees
    // (0 is horizontal, 90 vertical;
    // must be in this interval for source terms to work correctly)
    let wave_angle_deg = 60.;
    let wave_angle = wave_angle_deg * TAU / 360.;
    let wave_dir = dex::Vec2::new(wave_angle.cos(), wave_angle.sin());

    let pressure_wavenumber = 2.;
    let pressure_angular_vel = layers[0].p_wave_speed * pressure_wavenumber;
    let pressure_wave_vector = pressure_wavenumber * wave_dir;

    let shear_wavenumber = 2.;
    let shear_angular_vel = layers[0].s_wave_speed * shear_wavenumber;
    let shear_wave_vector = shear_wavenumber * wave_dir;

    // operators

    // spatially varying scaling factors
    let stiffness_scaling = mesh.scaling_dual(|s| {
        let l = layers.iter().find(|l| l.tris.contains(s.dual())).unwrap();
        l.stiffness
    });
    let density_scaling = mesh.scaling(|s| {
        let l = layers.iter().find(|l| l.edges.contains(s)).unwrap();
        1. / l.density
    });
    let mu_scaling = mesh.scaling_dual(|s| {
        let l = layers.iter().find(|l| l.tris.contains(s.dual())).unwrap();
        l.mu
    });

    let interp = dex::interpolate::dual_to_primal(&mesh);

    let ops = Ops {
        p_step: dt * stiffness_scaling * mesh.star() * mesh.d(),
        q_step: dt * density_scaling.clone() * mesh.star() * mesh.d(),
        // the interpolated operators have no effect everywhere except at material boundaries
        // (and break at the mesh boundary due to truncated dual cells)
        // so we can safely exclude the rest
        q_step_interp: (dt * density_scaling.clone() * mesh.d() * interp.clone())
            .exclude_subset(&layer_boundary_edges.complement(&mesh)),
        w_step: dt * mu_scaling * mesh.star() * mesh.d(),
        v_step: dt * density_scaling.clone() * mesh.star() * mesh.d(),
        v_step_interp: (dt * density_scaling * mesh.d() * interp.clone())
            .exclude_subset(&layer_boundary_edges.complement(&mesh)),
    };

    // source terms

    // sources are only active for one wave pulse
    let p_pulse_time = TAU / pressure_angular_vel;
    let s_pulse_time = TAU / shear_angular_vel;

    // farthest point to the right is the last point where the pulse is active;
    // use this to turn the bottom boundary into an absorbing boundary
    // after the pulse has been sent
    let far_end = mesh
        .simplices_in(&bottom_edges)
        .map(|e| e.vertices().max_by(|a, b| a.x.total_cmp(&b.x)).unwrap())
        .max_by(|a, b| a.x.total_cmp(&b.x))
        .unwrap();

    // for angled pulses we need to compute when it reaches the given point
    // to avoid discontinuities
    let p_pulse_active = |pos: dex::Vec2, t: f64| -> bool {
        let start_time = wave_dir.dot(&pos) / layers[0].p_wave_speed;
        let end_time = start_time + p_pulse_time;
        t >= start_time && t <= end_time
    };
    let p_pulse_active_anywhere = |t: f64| -> bool {
        if !send_p_pulse {
            return false;
        }
        t <= (wave_dir.dot(&far_end) / layers[0].p_wave_speed) + p_pulse_time
    };
    let s_pulse_active = |pos: dex::Vec2, t: f64| -> bool {
        let start_time = wave_dir.dot(&pos) / layers[0].s_wave_speed;
        let end_time = start_time + s_pulse_time;
        t >= start_time && t <= end_time
    };
    let s_pulse_active_anywhere = |t: f64| -> bool {
        if !send_s_pulse {
            return false;
        }
        t <= (wave_dir.dot(&far_end) / layers[0].s_wave_speed) + s_pulse_time
    };

    let pressure_source_vec = |pos: dex::Vec2, dir: dex::UnitVec2, t: f64| -> f64 {
        if !p_pulse_active(pos, t) {
            return 0.;
        }
        let normal = dex::Vec2::new(dir.y, -dir.x);
        let vel = -pressure_wave_vector
            * f64::sin(pressure_angular_vel * t - pressure_wave_vector.dot(&pos));
        vel.dot(&normal)
    };
    let shear_source_vec = |pos: dex::Vec2, dir: dex::UnitVec2, t: f64| -> f64 {
        if !s_pulse_active(pos, t) {
            return 0.;
        }
        let normal = dex::Vec2::new(dir.y, -dir.x);
        let vel =
            -shear_wave_vector * f64::sin(shear_angular_vel * t - shear_wave_vector.dot(&pos));
        vel.dot(&normal)
    };

    // run simulation

    let state = State {
        t: 0.,
        p: mesh.new_zero_cochain(),
        q: mesh.new_zero_cochain(),
        w: mesh.new_zero_cochain(),
        v: mesh.new_zero_cochain(),
        draw_pressure: false,
        draw_arrows: false,
    };

    let initial_state = state.clone();

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-0.0..1.0),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            state.q += &ops.q_step * &state.p + &ops.q_step_interp * &state.w;
            state.v += &ops.v_step * &state.w + &ops.v_step_interp * &state.p;

            // sources applied to the flux and velocity vectors
            if p_pulse_active_anywhere(state.t) {
                mesh.integrate_overwrite(
                    &mut state.q,
                    &bottom_edges,
                    dex::quadrature::GaussLegendre6(|p, d| pressure_source_vec(p, d, state.t)),
                );
            }
            if s_pulse_active_anywhere(state.t) {
                mesh.integrate_overwrite(
                    &mut state.v,
                    &bottom_edges,
                    dex::quadrature::GaussLegendre6(|p, d| shear_source_vec(p, d, state.t)),
                );
            }

            // absorbing boundary
            for layer in &layers {
                let edges_here = boundary_edges.intersection(&layer.edges);

                for edge in mesh.simplices_in(&edges_here) {
                    let length = edge.volume();
                    // pressure from the adjacent dual vertex
                    let (orientation, tri) = edge.coboundary().next().unwrap();
                    // bottom boundary only becomes absorbing once the pulse has been sent
                    if !bottom_edges.contains(edge) || !p_pulse_active_anywhere(state.t) {
                        state.q[edge] = -state.p[tri.dual()] * length * orientation as f64
                            / (layer.p_wave_speed * layer.density);
                    }
                    if !bottom_edges.contains(edge) || !s_pulse_active_anywhere(state.t) {
                        state.v[edge] = -state.w[tri.dual()] * length * orientation as f64
                            / (layer.s_wave_speed * layer.density);
                    }
                }
            }

            state.p += &ops.p_step * &state.q;
            state.w += &ops.w_step * &state.v;

            state.t += dt;
        },
        draw: |state, draw| {
            draw.axes_2d(dv::AxesParams::default());

            if state.draw_pressure {
                draw.triangle_colors_dual(&state.p);
            } else {
                draw.triangle_colors_dual(&state.w);
            }

            let arst =
                mesh.integrate_cochain(dex::quadrature::Pointwise(|p| p[0] * p[1] / (PI * PI)));
            let inted = &interp * &arst;
            draw.vertex_colors(&inted);

            draw.wireframe(dv::WireframeParams {
                width: dv::LineWidth::ScreenPixels(1.),
                ..Default::default()
            });
            for (idx, layer) in layers.iter().enumerate() {
                // layer boundaries with thicker lines
                draw.wireframe_subset(
                    dv::WireframeParams {
                        width: dv::LineWidth::ScreenPixels(3.),
                        ..Default::default()
                    },
                    &layer.boundary,
                );

                // text for material parameters
                let layer_height = TAU / layers.len() as f64;
                let lambda = layer.stiffness - 2. * layer.mu;
                draw.text(dv::TextParams {
                    text: &format!("λ: {}\nμ: {}\nρ: {}", lambda, layer.mu, layer.density),
                    position: dex::Vec2::new(TAU + 0.1, (idx as f64 + 0.5) * layer_height),
                    anchor: dv::TextAnchor::MidLeft,
                    font_size: 20.,
                    line_height: 24.,
                    ..Default::default()
                });
            }

            if state.draw_arrows {
                if state.draw_pressure {
                    draw.flux_arrows(&state.q, dv::ArrowParams::default());
                } else {
                    draw.velocity_arrows(&state.v, dv::ArrowParams::default());
                }
            }

            draw.text(dv::TextParams {
                text: if state.draw_pressure {
                    "Displaying pressure"
                } else {
                    "Displaying shear"
                },
                position: dex::Vec2::new(PI, 2. * PI),
                anchor: dv::TextAnchor::BottomMid,
                font_size: 24.,
                ..Default::default()
            });
        },
        on_key: |key, state| match key {
            dv::KeyCode::KeyP => {
                state.draw_pressure = !state.draw_pressure;
            }
            dv::KeyCode::KeyA => {
                state.draw_arrows = !state.draw_arrows;
            }
            dv::KeyCode::KeyR => {
                *state = initial_state.clone();
            }
            _ => {}
        },
    })?;

    Ok(())
}
