//! Example simulation of a vibrating membrane with fixed boundary.
//!
//! The acoustic wave equation formulated for the velocity potential ɸ,
//! ∂^2ɸ/∂t^2 - c^2 ∇^2 ɸ = 0,
//! is split into a system of scalar pressure p and vector velocity v,
//! ∂p/∂t - c^2 ∇·v = 0
//! and ∂v/∂t - ∇p = 0.
//! These are expressed in exterior calculus as
//! ∂p/∂t - c^2 ★d★v = 0
//! and ∂v/∂t - ★dp = 0
//! where p is a 0-form and v as 1-form.
//! p is expressed as a primal 0-cochain
//! and v as a primal 1-cochain.
//! Time discretization is performed with leapfrog integration.
//! A Dirichlet condition `ɸ = 0` is imposed on the boundary.
//!
//! A more thorough explanation of the mathematical model
//! can be found in the author's master's thesis,
//! http://urn.fi/URN:NBN:fi:jyu-202310035379
//! where this and other acoustics examples in this repo were originally created.

use dex::visuals as dv;
use dexterior as dex;

type Pressure = dex::Cochain<0, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;

#[derive(Clone, Debug)]
struct State {
    p: Pressure,
    v: Velocity,
}

impl dv::AnimationState for State {
    fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
        Self {
            p: old.p.lerp(&new.p, t),
            v: old.v.lerp(&new.v, t),
        }
    }
}

struct Ops {
    p_step: dex::Op<Velocity, Pressure>,
    v_step: dex::Op<Pressure, Velocity>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let msh_bytes = include_bytes!("./meshes/2d_square_pi_x_pi.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    let dt = 1. / 10.;
    let wave_speed_sq = 1.0f64.powi(2);

    let ops = Ops {
        p_step: (-dt * wave_speed_sq * mesh.star() * mesh.d() * mesh.star())
            // Dirichlet boundary implemented by removing rows from the operator
            .exclude_subset(&mesh.boundary()),
        v_step: dt * mesh.d(),
    };

    let state = State {
        // this is zero everywhere on the boundary of the [0, pi] x [0, pi] square
        // (fulfilling the Dirichlet condition)
        // as long as the coefficients on v[0].x and v[0].y are integers
        p: mesh.integrate_cochain(dex::quadrature::Pointwise(|v| {
            // something funky about nalgebra types prevents us from using v.x and v.y here :/
            f64::sin(3.0 * v[0]) * f64::sin(2.0 * v[1])
        })),
        v: mesh.new_zero_cochain(),
    };

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-1.0..1.0),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            state.p += &ops.p_step * &state.v;
            state.v += &ops.v_step * &state.p;
        },
        draw: |state, draw| {
            draw.vertex_colors(&state.p);
            draw.wireframe(dv::WireframeParams::default());
            draw.velocity_arrows(&state.v, dv::ArrowParams::default());
            draw.axes_2d(dv::AxesParams::default());

            draw.text(dv::TextParams {
                text: "test text\nχξΔ",
                position: mesh.vertices()[2],
                anchor: dv::TextAnchor::BottomMid,
                color: dv::TextColor::rgb(150, 0, 0),
                attrs: dv::glyphon::Attrs::new().style(dv::glyphon::Style::Italic),
                ..Default::default()
            });
        },
        on_key: |_, _| {},
    })?;

    Ok(())
}
