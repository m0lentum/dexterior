//! Cochains, i.e. values assigned to elements of a mesh.

use nalgebra as na;
use typenum as tn;

/// Marker type indicating a [`Cochain`][self::Cochain]
/// corresponds to a primal mesh.
#[derive(Clone, Copy, Debug)]
pub struct Primal;
/// Marker type indicating a [`Cochain`][self::Cochain]
/// corresponds to a dual mesh.
#[derive(Clone, Copy, Debug)]
pub struct Dual;

pub trait MeshPrimality {
    type Opposite: MeshPrimality;
}
impl MeshPrimality for Primal {
    type Opposite = Dual;
}
impl MeshPrimality for Dual {
    type Opposite = Primal;
}

/// A vector of values corresponding to
/// a set of `Dimension`-dimensional cells on a mesh.
///
/// TODOC: once the API is confirmed to actually work,
/// write about how typenum is used here
#[derive(Clone)]
pub struct Cochain<Dimension, Primality> {
    pub values: na::DVector<f64>,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
}

impl<Dimension, Primality> Cochain<Dimension, Primality> {
    // constructors only exposed to crate
    // because cochains are always based on a mesh
    // and it doesn't make sense for a user to create them directly;
    // public constructors are methods on SimplicialMesh

    pub(crate) fn zeros(len: usize) -> Self {
        Self {
            values: na::DVector::zeros(len),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Dimension, Primality> std::fmt::Debug for Cochain<Dimension, Primality>
where
    Dimension: tn::Unsigned,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dim = Dimension::to_usize();
        write!(f, "{}-cochain, values {:?}", dim, self.values)
    }
}
