//! This is the core crate containing most of `dexterior`'s functionality
//! (specifically, everything except the visuals).
//! See the `dexterior` crate's documentation for an in-depth introduction.

#![warn(missing_docs)]

pub mod mesh;
#[doc(inline)]
pub use mesh::{
    Dual, DualCellIter, DualCellView, Primal, SimplexIter, SimplexView, SimplicialMesh,
};

pub mod cochain;
#[doc(inline)]
pub use cochain::Cochain;

pub mod operator;
#[doc(inline)]
pub use operator::{ComposedOperator, ExteriorDerivative, HodgeStar, Op, Operator};

pub mod gmsh;
