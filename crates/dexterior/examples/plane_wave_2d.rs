//! Example simulation of an acoustic plane wave
//! ɸ = -cos(ωt - κ·x) propagating through space.
//!
//! The equations are the same as in `membrane.rs`,
//! but instead of the 1-form velocity v we use the 1-form flux q = ★v.
//! q is expressed as a primal 1-cochain, and p as a dual 0-cochain
//! (this difference isn't actually needed here,
//! but becomes relevant in flux-based absorbing boundary conditions).
//!
//! The wave is set on the boundary as a Dirichlet condition
//! and simulated inside the domain.
//! This can be used to measure the accuracy of the method,
//! since the exact solution ɸ is known.

use dex::visuals as dv;
use dexterior as dex;

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
    let wave_dir = dex::Unit::new_normalize(dex::Vec2::new(1., 1.));
    let wavenumber = 2.;
    let wave_vector = wavenumber * *wave_dir;
    let wave_angular_vel = wavenumber * wave_speed;

    let eval_wave_pressure = |t: f64, pos: dex::Vec2| -> f64 {
        wave_angular_vel * f64::sin(wave_angular_vel * t - wave_vector.dot(&pos))
    };
    let eval_wave_flux = |t: f64, pos: dex::Vec2, dir: dex::UnitVec2| -> f64 {
        let normal = dex::Vec2::new(dir.y, -dir.x);
        let vel = -wave_vector * f64::sin(wave_angular_vel * t - wave_vector.dot(&pos));
        vel.dot(&normal)
    };

    let ops = Ops {
        p_step: dt * wave_speed.powi(2) * mesh.star() * mesh.d(),
        q_step: dt * mesh.star() * mesh.d(),
    };

    let state = State {
        p: mesh.integrate_cochain(dex::quadrature::Pointwise(|v| eval_wave_pressure(0., v))),
        q: mesh.integrate_cochain(dex::quadrature::GaussLegendre6(|v, d| {
            eval_wave_flux(dt / 2., v, d)
        })),
        t: 0.,
    };

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-2.0..2.0),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            // q is computed at a time instance offset by half dt (leapfrog integration)
            let t_at_q = state.t + dt / 2.;
            state.t += dt;

            state.q += &ops.q_step * &state.p;
            let boundary_wave: Flux = mesh.integrate_cochain_partial(
                &mesh.boundary(),
                dex::quadrature::GaussLegendre6(|v, d| eval_wave_flux(t_at_q, v, d)),
            );
            for edge in mesh.simplices_in(&mesh.boundary()) {
                state.q[edge] = boundary_wave[edge];
            }

            state.p += &ops.p_step * &state.q;
        },
        draw: |state, draw| {
            draw.triangle_colors_dual(&state.p);
            draw.wireframe(dv::WireframeParams::default());
            draw.flux_arrows(
                &state.q,
                dv::ArrowParams {
                    scaling: dv::ArrowParams::default().scaling / 2.,
                    ..Default::default()
                },
            );
            draw.axes_2d(dv::AxesParams::default());
        },
        on_key: |_, _| {},
    })?;

    Ok(())
}
