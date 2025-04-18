//! Example simulation of an acoustic plane wave
//! ɸ = -cos(ωt - κ·x) being scattered by an obstacle.
//!
//! The wave is applied on the boundary of the obstacle
//! as a Dirichlet condition similarly to plane_wave_2d.rs.
//! On the outer boundary an absorbing boundary condition
//! is used to simulate the wave leaving the domain unimpeded.
//!
//! This one uses a higher-resolution mesh than the other acoustics examples
//! and demonstrates an alternative material parameterization
//! with wave speed expressed in terms of stiffness and density.

use dex::visuals as dv;
use dexterior as dex;
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

    let dt = 1. / 90.;
    let stiffness = 32.;
    let density = 2.;
    let wave_speed = f64::sqrt(stiffness / density);

    let wave_dir = dex::Unit::new_normalize(dex::Vec2::new(0., 1.));
    let wavenumber = 2.;
    let wave_vector = wavenumber * *wave_dir;
    let wave_angular_vel = wavenumber * wave_speed;
    let wave_period = TAU / wave_angular_vel;
    // would be nice to have some of these parameters taken from the command line
    let show_inc_wave = true;

    let eval_wave_pressure = |t: f64, pos: dex::Vec2| -> f64 {
        wave_angular_vel * f64::sin(wave_angular_vel * t - wave_vector.dot(&pos))
    };
    let eval_wave_flux = |t: f64, pos: dex::Vec2, dir: dex::UnitVec2| -> f64 {
        let normal = dex::Vec2::new(dir.y, -dir.x);
        let vel = -wave_vector * f64::sin(wave_angular_vel * t - wave_vector.dot(&pos));
        vel.dot(&normal)
    };

    let ops = Ops {
        p_step: dt * stiffness * mesh.star() * mesh.d(),
        q_step: (dt * density.recip() * mesh.star() * mesh.d())
            .exclude_subset(&absorbing_boundary)
            .exclude_subset(&scattering_boundary),
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
        let sin_val = f64::sin((t / transition_time) * (PI / 2.0));
        (2.0 - sin_val) * sin_val
    };

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-16.0..16.0),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            // q is computed at a time instance offset by half dt (leapfrog integration)
            let t_at_q = state.t + dt / 2.;
            state.t += dt;

            state.q += &ops.q_step * &state.p;

            // scatterer boundary

            let easing_coef = easing(t_at_q);
            mesh.integrate_overwrite(
                &mut state.q,
                &scattering_boundary,
                dex::quadrature::GaussLegendre6(|v, d| easing_coef * eval_wave_flux(t_at_q, v, d)),
            );

            // absorbing boundary

            let denom = wave_speed * density;
            for edge in mesh.simplices_in(&absorbing_boundary) {
                let length = edge.volume();
                // edges on the boundary always only border one volume element,
                // and the adjacent dual vertex is the one corresponding to that element
                let (orientation, coboundary) = edge.coboundary().next().unwrap();
                state.q[edge] = -state.p[coboundary.dual()] * length * orientation as f64 / denom;
            }

            state.p += &ops.p_step * &state.q;
        },
        draw: |state, draw| {
            // add the incident wave (which we don't simulate) to the visualization
            let total_p = if show_inc_wave {
                let inc_p = mesh.integrate_cochain(dex::quadrature::Pointwise(|v| {
                    eval_wave_pressure(state.t, v)
                }));
                &state.p + inc_p
            } else {
                state.p.clone()
            };

            draw.triangle_colors_dual(&total_p);

            draw.wireframe_subset(
                dv::WireframeParams {
                    width: dv::LineWidth::ScreenPixels(3.),
                    color: dv::palette::named::DARKSLATEBLUE.into(),
                },
                &absorbing_boundary,
            );
            draw.wireframe_subset(
                dv::WireframeParams {
                    width: dv::LineWidth::ScreenPixels(3.),
                    color: dv::palette::named::DARKRED.into(),
                },
                &scattering_boundary,
            );
            draw.axes_2d(dv::AxesParams {
                minor_ticks: 4,
                // TODO: the defaults mixing worldspace length and pixel width
                // doesn't scale well in a number of situations,
                // they should either be all pixels or all worldspace.
                // both have tradeoffs, worldspace sizes respond well to window size changes
                // but don't do well with different mesh sizes,
                // whereas pixel sizes are the opposite
                tick_length: 0.2,
                tick_interval: 4.,
                width: dv::LineWidth::ScreenPixels(3.),
                // also TODO: this is a large mesh where padding in worldspace
                // doesn't work very well;
                // padding should instead be scaled in terms of pixels
                padding: 1.,
                ..Default::default()
            });
        },
        on_key: |_, _| {},
    })?;

    Ok(())
}
