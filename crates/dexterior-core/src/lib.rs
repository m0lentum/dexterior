//! This is the core crate containing most of `dexterior`'s functionality
//! (specifically, everything except the visuals).
//! See the `dexterior` crate's documentation for an in-depth introduction.

#![warn(missing_docs)]

pub mod mesh;
#[doc(inline)]
pub use mesh::{
    Dual, DualCellIter, DualCellView, Primal, SimplexIter, SimplexView, SimplicialMesh, Subset,
};

pub mod cochain;
#[doc(inline)]
pub use cochain::Cochain;

pub mod operator;
#[doc(inline)]
pub use operator::{ComposedOperator, ExteriorDerivative, HodgeStar, Op, Operator};

pub mod gmsh;

pub mod interpolate;

pub mod quadrature;

pub(crate) mod permutation;

// nalgebra re-exports of common types for convenience

pub use nalgebra as na;
/// Type alias for a 2D `nalgebra` vector.
pub type Vec2 = na::Vector2<f64>;
/// Type alias for a 2D `nalgebra` unit vector.
pub type UnitVec2 = na::Unit<Vec2>;
/// Type alias for a 3D `nalgebra` vector.
pub type Vec3 = na::Vector3<f64>;
/// Type alias for a 3D `nalgebra` unit vector.
pub type UnitVec3 = na::Unit<Vec3>;
/// Type alias for a general `nalgebra` unit vector.
pub type Unit<T> = na::Unit<T>;
