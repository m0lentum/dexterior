//! Cochains, i.e. values assigned to elements of a mesh.

use nalgebra as na;

/// A vector of values corresponding to
/// a set of `Dimension`-dimensional cells on a mesh.
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

impl<Dimension, Primality> crate::operator::OperatorInput for Cochain<Dimension, Primality> {
    fn values(&self) -> &na::DVector<f64> {
        &self.values
    }

    fn from_values(values: na::DVector<f64>) -> Self {
        Self {
            values,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Dimension, Primality> std::fmt::Debug for Cochain<Dimension, Primality>
where
    Dimension: na::DimName,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-cochain, values {:?}", Dimension::USIZE, self.values)
    }
}
