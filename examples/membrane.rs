//! Example simulation of a vibrating membrane with fixed boundary.
//! Acoustic wave equation, Dirichlet boundary `ɸ = 0`.
//!
//! This is very incomplete at the moment,
//! made to point out unimplemented parts of the library.

use dexterior as dex;

type Pressure = dex::Cochain<0, dex::Dual>;
type Flux = dex::Cochain<1, dex::Primal>;

struct State {
    p: Pressure,
    q: Flux,
}

struct Ops {
    p_step: Box<dyn dex::Operator<Input = Flux, Output = Pressure>>,
    q_step: Box<dyn dex::Operator<Input = Pressure, Output = Flux>>,
}

fn main() {
    // TODO: generate a mesh with desired dimensions
    let mesh = dex::simplicial_complex::tiny_mesh_2d();

    // TODO: pick dt based on minimum edge length
    let dt = 0.17;
    let wave_speed = 1.0;

    // TODO: methods to easily create a nonzero initial state
    let mut state = State {
        p: mesh.new_zero_cochain(),
        q: mesh.new_zero_cochain(),
    };

    // TODO: implement star correctly
    //
    // TODO: allow multiplying operators by scalars
    //
    // TODO: boundary conditions:
    // operators need to be able to have rows removed
    // and replaced based on which simplices they correspond to.
    // this requires some kind of tagging system and APIs for modifying operators
    let ops = Ops {
        p_step: Box::new(mesh.star() * mesh.d()),
        q_step: Box::new(mesh.star() * mesh.d()),
    };

    let step_count = 60;
    for _ in 0..step_count {
        state.p += &ops.p_step * &state.q;
        state.q += &ops.q_step * &state.p;
    }

    // TODO: visualize this somehow
    // (maybe matplotlib for now, custom real-time renderer later)
}
