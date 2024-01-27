//! Example simulation of an acoustic plane wave
//! ɸ = -cos(ωt - κ·x) propagating through space.
//! The wave is set on the boundary as a Dirichlet condition
//! and simulated inside the domain.
//! This can be used to measure the accuracy of the method,
//! since the exact solution is known.

use dexterior as dex;
use dexterior_visuals as dv;
use nalgebra as na;

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
    let msh_bytes = include_bytes!("./meshes/2d_square_pi_x_pi.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    let dt = 1. / 60.;
    let wave_speed = 1f64;
    let wave_dir = na::Unit::new_normalize(na::Vector2::new(1., 1.));
    let wavenumber = 2.;
    let wave_vector = wavenumber * *wave_dir;
    let wave_angular_vel = wavenumber * wave_speed;

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
        q_step: dt * mesh.star() * mesh.d(),
    };

    let state = State {
        p: mesh.integrate_cochain(|v| eval_wave_pressure(0., &v[0])),
        q: mesh.integrate_cochain(|v| eval_wave_flux(dt / 2., v)),
        t: 0.,
    };

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams::default(),
        dt,
        state,
        step: |state| {
            // q is computed at a time instance offset by half dt (leapfrog integration)
            let t_at_q = state.t + dt / 2.;
            state.t += dt;

            state.q += &ops.q_step * &state.p;
            // TODO: option to integrate only a subset of the cochain
            let boundary_wave: Flux = mesh.integrate_cochain(|v| eval_wave_flux(t_at_q, v));
            for idx in mesh.boundary::<1>().indices.ones() {
                state.q.values[idx] = boundary_wave.values[idx];
            }

            state.p += &ops.p_step * &state.q;
        },
        draw: |state, draw| {
            // TODO: draw pressure as solid colored triangles
            draw.wireframe();
            draw.flux_arrows(&state.q, dv::ArrowParameters::default());
            draw.axes_2d(dv::AxesParameters::default());
        },
    });

    Ok(())
}
