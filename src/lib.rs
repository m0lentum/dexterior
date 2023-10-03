pub mod mesh;
pub use mesh::{Dual, Primal, SimplexIter, SimplexView, SimplexViewMut, SimplicialMesh};

pub mod cochain;
pub use cochain::Cochain;

pub mod operator;
pub use operator::{ComposedOperator, ExteriorDerivative, HodgeStar};
