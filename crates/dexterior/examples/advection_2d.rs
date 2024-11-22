//! Example of advecting cochains along a constant velocity field in 2D.

use dex::visuals as dv;
use dexterior as dex;
use nalgebra as na;

use std::f64::consts::PI;

#[derive(Clone, Debug)]
struct State {
    zero: dex::Cochain<0, dex::Dual>,
}

impl dv::AnimationState for State {
    fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
        Self {
            zero: old.zero.lerp(&new.zero, t),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let msh_bytes = include_bytes!("./meshes/2d_square_pi_x_pi.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    let dt = 0.02;

    let state = State {
        // start with a circular area centered near the bottom edge
        zero: mesh.integrate_cochain(dex::quadrature::Pointwise(|v| {
            f64::max(0., 0.5 - (v - na::Vector2::new(PI / 2., 1.)).magnitude()).sqrt()
        })),
    };

    // velocity vector field rotates around the center
    let velocity_field: dex::Cochain<1, dex::Primal> =
        mesh.integrate_cochain(dex::quadrature::GaussLegendre6(|v, d| {
            let dist_from_center = v - na::Vector2::new(PI / 2., PI / 2.);
            if dist_from_center.magnitude() > PI / 3. {
                0.
            } else {
                let left_normal = na::Vector2::new(-dist_from_center.y, dist_from_center.x);
                0.5 * left_normal.dot(&d)
            }
        }));

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-0.5..1.0),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            // Lie derivative of a k-form expressed in terms of the interior product i_X
            // as L_X = i_Xdω + di_Xω
            // which in turn is expressed as i_X(ω) = (-1)^{k(n-k)} ★(★ω ⋀ X󰽫)
            // for a 0-form only the first term i_Xdω exists
            let star_d_zero = mesh.star() * mesh.d() * &state.zero;
            let lie_d = mesh.star() * &mesh.wedge(&star_d_zero, &velocity_field);
            state.zero -= dt * lie_d;
        },
        draw: |state, draw| {
            draw.triangle_colors_dual(&state.zero);
            draw.wireframe(dv::WireframeParams::default());
            draw.axes_2d(dv::AxesParams::default());
            draw.velocity_arrows(&velocity_field, dv::ArrowParams::default());
        },
    })?;

    Ok(())
}
