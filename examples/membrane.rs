//! Example simulation of a vibrating membrane with fixed boundary.
//! Acoustic wave equation, Dirichlet boundary `ɸ = 0`.

use dexterior as dex;
use dexterior_visuals as dv;
use nalgebra as na;

type Pressure = dex::Cochain<0, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;

#[derive(Clone, Debug)]
struct State {
    p: Pressure,
    v: Velocity,
}

struct Ops {
    p_step: dex::Op<Velocity, Pressure>,
    v_step: dex::Op<Pressure, Velocity>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let msh_bytes = include_bytes!("./meshes/2d_square_pi_x_pi.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    // TODO: pick dt based on minimum edge length
    // (this requires an iterator with access to primal volumes
    // and is currently absurdly low
    // because I haven't implemented timing control for the real-time visualization yet
    // so it runs way too fast :^)
    let dt = 1.0 / 2000.0;
    let wave_speed_sq = 1.0f64.powi(2);

    let ops = Ops {
        p_step: (-dt * wave_speed_sq * mesh.star() * mesh.d() * mesh.star())
            // Dirichlet boundary implemented by removing rows from the operator
            .exclude_subset(mesh.boundary()),
        v_step: mesh.d().into(),
    };

    let mut state = State {
        // this is zero everywhere on the boundary of the [0, pi] x [0, pi] square
        // as long as the coefficients on v[0].x and v[0].y are integers
        p: mesh.integrate_cochain(|v| f64::sin(3.0 * v[0].x) * f64::sin(2.0 * v[0].y)),
        v: mesh.new_zero_cochain(),
    };

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-1.0..1.0),
            ..Default::default()
        },
        step: |draw| {
            state.p += &ops.p_step * &state.v;
            state.v += &ops.v_step * &state.p;

            draw.vertex_colors(&state.p);
            draw.wireframe();
            // just testing
            draw.line_strip(&[
                na::Vector3::new(2.5, 0.5, 0.0),
                na::Vector3::new(3.5, 1.0, 0.0),
                na::Vector3::new(2.5, 1.0, 0.0),
                na::Vector3::new(3.5, 1.5, 0.0),
                na::Vector3::new(3.0, 3.0, 0.0),
            ]);
            draw.line_list(&[
                na::Vector3::new(0.5, 0.5, 0.0),
                na::Vector3::new(1.5, 1.0, 0.0),
                na::Vector3::new(0.5, 1.0, 0.0),
                na::Vector3::new(1.5, 1.5, 0.0),
            ]);
            draw.axes_2d();
        },
    });

    Ok(())
}
