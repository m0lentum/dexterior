//! Example simulation of an acoustic plane wave
//! ɸ = -cos(ωt - κ·x) being scattered by an obstacle.
//!
//! The wave is applied on the boundary of the obstacle
//! as a Dirichlet condition similarly to plane_wave_2d.rs.
//! On the outer boundary an absorbing boundary condition
//! is used to simulate the wave leaving the domain unimpeded.

use dex::visuals as dv;
use dexterior as dex;
use nalgebra as na;
use std::f64::consts::{PI, TAU};

type Pressure = dex::Cochain<0, dex::Dual>;
type Flux = dex::Cochain<1, dex::Primal>;

#[derive(Clone, Debug)]
struct State {
    p: Pressure,
    q: Flux,
    t: f64,
}

impl dv::AnimationState for State {
    fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
        Self {
            p: old.p.lerp(&new.p, t),
            q: old.q.lerp(&new.q, t),
            t: old.t + t * (new.t - old.t),
        }
    }
}

struct Ops {
    p_step: dex::Op<Flux, Pressure>,
    q_step: dex::Op<Pressure, Flux>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let msh_bytes = include_bytes!("./meshes/2d_star.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    // named physical groups not supported in the gmsh module yet,
    // so we need to use the integer tags
    let absorbing_boundary = mesh.get_subset::<1>("10").expect("Subset not found");
    let scattering_boundary = mesh.get_subset::<1>("100").expect("Subset not found");

    let dt = 1. / 20.;
    let wave_speed = 1f64;
    let wave_dir = na::Unit::new_normalize(na::Vector2::new(0., 1.));
    let wavenumber = 2.;
    let wave_vector = wavenumber * *wave_dir;
    let wave_angular_vel = wavenumber * wave_speed;
    let wave_period = TAU / wave_angular_vel;
    // would be nice to have some of these parameters taken from the command line
    let show_inc_wave = true;

    let eval_wave_pressure = |t: f64, pos: &na::Vector2<f64>| -> f64 {
        wave_angular_vel * f64::sin(wave_angular_vel * t - wave_vector.dot(pos))
    };
    // analytic form of the flux of the wave integrated over a mesh edge.
    // TODO: add numerical integration quadratures to do this more easily
    let eval_wave_flux = |t: f64, p: &[na::Vector2<f64>]| -> f64 {
        let kdotp = wave_vector.dot(&p[0]);
        let l = p[1] - p[0];
        let kdotl = wave_vector.dot(&l);
        let kdotn = wave_vector.dot(&na::Vector2::new(l.y, -l.x));
        let wave_angle = wave_angular_vel * t;
        if kdotl.abs() < 1e-5 {
            -kdotn * f64::sin(wave_angle - kdotp)
        } else {
            (kdotn / kdotl) * (f64::cos(wave_angle - kdotp) - f64::cos(wave_angle - kdotp - kdotl))
        }
    };

    let ops = Ops {
        p_step: dt * wave_speed.powi(2) * mesh.star() * mesh.d(),
        q_step: (dt * mesh.star() * mesh.d())
            .exclude_subset(absorbing_boundary)
            .exclude_subset(scattering_boundary),
    };

    // introduce source terms with easing starting from zero ("Mur transition")
    // for a smooth start without breaking boundary conditions
    let state = State {
        p: mesh.new_zero_cochain(),
        q: mesh.new_zero_cochain(),
        t: 0.,
    };

    let transition_time = wave_period;
    let easing = |t: f64| -> f64 {
        if t >= transition_time {
            return 1.;
        }
        let sin_val = f64::sin((t / transition_time) * (PI / 2.0));
        (2.0 - sin_val) * sin_val
    };

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-4.0..4.0),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            // q is computed at a time instance offset by half dt (leapfrog integration)
            let t_at_q = state.t + dt / 2.;
            state.t += dt;

            state.q += &ops.q_step * &state.p;
            // TODO: option to integrate only a subset of the cochain
            let boundary_wave: Flux = mesh.integrate_cochain(|v| eval_wave_flux(t_at_q, v));
            let easing_coef = easing(t_at_q);
            for idx in scattering_boundary.indices.ones() {
                state.q.values[idx] = easing_coef * boundary_wave.values[idx];
            }

            // absorbing boundary.
            // TODO: this could be done as an operator
            // if we had a way to make custom operators
            for idx in absorbing_boundary.indices.ones() {
                let edge = mesh.get_simplex_by_index::<1>(idx);
                let length = edge.volume();
                // edges on the boundary always only border one volume element,
                // and the adjacent dual vertex is the one corresponding to that element
                let (orientation, coboundary) = edge.coboundary().next().unwrap();
                state.q.values[idx] =
                    -state.p.values[coboundary.index()] * length * orientation as f64;
            }

            state.p += &ops.p_step * &state.q;
        },
        draw: |state, draw| {
            // add the incident wave (which we don't simulate) to the visualization
            // TODO: having to use references for the second operand here
            // is annoying especially if you wanted to multiply with a scalar too,
            // and the errors for not doing so are confusing;
            // we should add an Add impl for owned cochains
            let (total_p, total_q) = if show_inc_wave {
                (
                    &(&state.p + &mesh.integrate_cochain(|v| eval_wave_pressure(state.t, &v[0]))),
                    &(&state.q + &mesh.integrate_cochain(|v| eval_wave_flux(state.t, v))),
                )
            } else {
                (&state.p, &state.q)
            };

            draw.triangle_colors_dual(total_p);
            draw.wireframe(dv::WireframeParams {
                width: dv::LineWidth::WorldUnits(0.015),
                ..Default::default()
            });
            draw.flux_arrows(
                total_q,
                dv::ArrowParams {
                    scaling: dv::ArrowParams::default().scaling / 2.,
                    width: dv::LineWidth::WorldUnits(0.015),
                    ..Default::default()
                },
            );
            draw.axes_2d(dv::AxesParams {
                minor_ticks: 4,
                // TODO: the defaults mixing worldspace length and pixel width
                // doesn't scale well in a number of situations,
                // they should either be all pixels or all worldspace.
                // both have tradeoffs, worldspace sizes respond well to window size changes
                // but don't do well with different mesh sizes,
                // whereas pixel sizes are the opposite
                tick_length: 0.075,
                width: dv::LineWidth::WorldUnits(0.02),
                ..Default::default()
            });
        },
    })?;

    Ok(())
}
