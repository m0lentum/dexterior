//! Example simulation of a vibrating membrane with fixed boundary.
//! Acoustic wave equation, Dirichlet boundary `ɸ = 0`.
//!
//! This is very incomplete at the moment,
//! made to point out unimplemented parts of the library.

use dexterior as dex;

type Pressure = dex::Cochain<0, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;

#[derive(Clone, Debug)]
struct State {
    p: Pressure,
    v: Velocity,
}

struct Ops {
    p_step: Box<dyn dex::Operator<Input = Velocity, Output = Pressure>>,
    v_step: Box<dyn dex::Operator<Input = Pressure, Output = Velocity>>,
}

fn main() {
    // TODO: generate a mesh with desired dimensions
    let mesh = dex::mesh::tiny_mesh_2d();

    // TODO: pick dt based on minimum edge length
    // (this requires an iterator with access to primal volumes)
    let dt = 0.17;
    let wave_speed_sq = 1.0f64.powi(2);

    let mut state = State {
        // note: this initial state currently doesn't fulfill the boundary condition
        // because the shape of the mesh is not what this experiment was designed for.
        // once we can generate meshes it should be a [0, 0] x [pi, pi] square,
        // where this is zero everywhere on the boundary.
        p: mesh.integrate_cochain(|v| f64::sin(3.0 * v[0].x) * f64::sin(2.0 * v[0].y)),
        v: mesh.new_zero_cochain(),
    };

    let ops = Ops {
        p_step: Box::new(
            (dt * wave_speed_sq * mesh.star() * mesh.d() * mesh.star())
                // Dirichlet boundary implemented by removing rows from the operator
                .exclude_subset(mesh.boundary()),
        ),
        v_step: Box::new(mesh.d()),
    };

    let step_count = 60;
    for _ in 0..step_count {
        state.p += &ops.p_step * &state.v;
        state.v += &ops.v_step * &state.p;
    }

    // TODO: visualize this somehow
    // (maybe matplotlib for now, custom real-time renderer later)
}
