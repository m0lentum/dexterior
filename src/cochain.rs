//! Cochains, i.e. values assigned to elements of a mesh.

use nalgebra as na;

/// A vector of values corresponding to
/// a set of `k`-dimensional cells on a mesh.
///
/// Cochains can be constructed using the following methods
/// on [`SimplicialMesh`][crate::SimplicialMesh]:
/// - [`new_zero_cochain`][crate::SimplicialMesh::new_zero_cochain]
pub type Cochain<const DIM: usize, Primality> = CochainImpl<na::Const<DIM>, Primality>;

/// The cochain type used internally by Dexterior.
///
/// This type cannot use const generics because they cannot currently
/// do the compile-time generic arithmetic needed for operators.
/// Thus, the more convenient alias [`Cochain`][self::Cochain]
/// is preferred for public APIs.
#[derive(Clone)]
pub struct CochainImpl<Dimension, Primality> {
    /// The underlying vector of real values, exposed for convenience.
    ///
    /// Note that changing the dimension of this vector at runtime
    /// will cause a dimension mismatch with operators,
    /// leading to a panic when an operator is applied.
    /// Use with caution.
    pub values: na::DVector<f64>,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
}

impl<Dimension, Primality> CochainImpl<Dimension, Primality> {
    // constructors only exposed to crate
    // because cochains are always based on a mesh
    // and it doesn't make sense for a user to create them directly;
    // public constructors are methods on SimplicialMesh

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

impl<Dimension, Primality> crate::operator::OperatorInput for CochainImpl<Dimension, Primality> {
    type Dimension = Dimension;
    type Primality = Primality;

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

impl<D, P> std::fmt::Debug for CochainImpl<D, P>
where
    D: na::DimName,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-cochain, values {:?}", D::USIZE, self.values)
    }
}

impl<D, P> PartialEq for CochainImpl<D, P> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<D, P> std::ops::Add for CochainImpl<D, P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        CochainImpl::from_values(self.values + rhs.values)
    }
}

impl<D, P> std::ops::AddAssign for CochainImpl<D, P> {
    fn add_assign(&mut self, rhs: Self) {
        self.values += rhs.values;
    }
}

impl<D, P> std::ops::Mul<CochainImpl<D, P>> for f64 {
    type Output = CochainImpl<D, P>;

    fn mul(self, rhs: CochainImpl<D, P>) -> Self::Output {
        CochainImpl::from_values(self * rhs.values)
    }
}

impl<D, P> std::ops::Mul<&CochainImpl<D, P>> for f64 {
    type Output = CochainImpl<D, P>;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        CochainImpl::from_values(self * &rhs.values)
    }
}
