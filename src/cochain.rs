//! Cochains, i.e. values assigned to elements of a mesh.

use nalgebra as na;

/// A vector of values corresponding to
/// a set of `k`-dimensional cells on a mesh.
///
/// Cochains can be constructed using the following methods
/// on [`SimplicialComplex`][crate::SimplicialComplex]:
/// - [`new_zero_cochain`][crate::SimplicialComplex::new_zero_cochain]
#[derive(Clone)]
pub struct Cochain<Dimension, Primality> {
    /// The underlying vector of real values, exposed for convenience.
    ///
    /// Note that changing the dimension of this vector at runtime
    /// will cause a dimension mismatch with operators,
    /// leading to a panic when an operator is applied.
    /// Use with caution.
    pub values: na::DVector<f64>,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
}

impl<Dimension, Primality> Cochain<Dimension, Primality> {
    // constructors only exposed to crate
    // because cochains are always based on a mesh
    // and it doesn't make sense for a user to create them directly;
    // public constructors are methods on SimplicialComplex

    #[inline]
    pub(crate) fn from_values(values: na::DVector<f64>) -> Self {
        Self {
            values,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline]
    pub(crate) fn zeros(len: usize) -> Self {
        Self::from_values(na::DVector::zeros(len))
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

//
// std trait impls for math ops and such
//

impl<D, P> std::fmt::Debug for Cochain<D, P>
where
    D: na::DimName,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-cochain, values {:?}", D::USIZE, self.values)
    }
}

impl<D, P> PartialEq for Cochain<D, P> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<D, P> std::ops::Add for Cochain<D, P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Cochain::from_values(self.values + rhs.values)
    }
}

impl<D, P> std::ops::AddAssign for Cochain<D, P> {
    fn add_assign(&mut self, rhs: Self) {
        self.values += rhs.values;
    }
}

impl<D, P> std::ops::Mul<Cochain<D, P>> for f64 {
    type Output = Cochain<D, P>;

    fn mul(self, rhs: Cochain<D, P>) -> Self::Output {
        Cochain::from_values(self * rhs.values)
    }
}

impl<D, P> std::ops::Mul<&Cochain<D, P>> for f64 {
    type Output = Cochain<D, P>;

    fn mul(self, rhs: &Cochain<D, P>) -> Self::Output {
        Cochain::from_values(self * &rhs.values)
    }
}
