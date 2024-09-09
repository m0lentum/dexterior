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

    /// Linearly interpolate along the line from `self` to `end`.
    pub fn lerp(&self, end: &Self, t: f64) -> Self {
        self + &(t * (end - self))
    }
}

impl<Dimension, Primality> crate::operator::Operand for CochainImpl<Dimension, Primality> {
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

// std trait impls for math ops and such
// (several permutations needed to also work with references.
// maybe this could be shortened with macros,
// but I can't be bothered)

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

// Index with SimplexViews
// (there's no equivalent view for dual cells yet
// so this only works for primal simplices at the moment)

impl<'a, D: na::DimName, const MESH_DIM: usize> std::ops::Index<crate::SimplexView<'a, D, MESH_DIM>>
    for CochainImpl<D, crate::Primal>
{
    type Output = f64;

    fn index(&self, simplex: crate::SimplexView<'a, D, MESH_DIM>) -> &Self::Output {
        &self.values[simplex.index()]
    }
}

impl<'a, D: na::DimName, const MESH_DIM: usize>
    std::ops::IndexMut<crate::SimplexView<'a, D, MESH_DIM>> for CochainImpl<D, crate::Primal>
{
    fn index_mut(&mut self, simplex: crate::SimplexView<'a, D, MESH_DIM>) -> &mut Self::Output {
        &mut self.values[simplex.index()]
    }
}

// Add

impl<D, P> std::ops::Add for CochainImpl<D, P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        CochainImpl::from_values(self.values + rhs.values)
    }
}

impl<D, P> std::ops::Add<&CochainImpl<D, P>> for CochainImpl<D, P> {
    type Output = Self;

    fn add(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        CochainImpl::from_values(self.values + &rhs.values)
    }
}

impl<D, P> std::ops::Add<CochainImpl<D, P>> for &CochainImpl<D, P> {
    type Output = CochainImpl<D, P>;

    fn add(self, rhs: CochainImpl<D, P>) -> Self::Output {
        CochainImpl::from_values(&self.values + rhs.values)
    }
}

impl<D, P> std::ops::Add for &CochainImpl<D, P> {
    type Output = CochainImpl<D, P>;

    fn add(self, rhs: Self) -> Self::Output {
        CochainImpl::from_values(&self.values + &rhs.values)
    }
}

// AddAssign

impl<D, P> std::ops::AddAssign for CochainImpl<D, P> {
    fn add_assign(&mut self, rhs: Self) {
        self.values += rhs.values;
    }
}

impl<D, P> std::ops::AddAssign<&CochainImpl<D, P>> for CochainImpl<D, P> {
    fn add_assign(&mut self, rhs: &CochainImpl<D, P>) {
        self.values += &rhs.values;
    }
}

// Neg

impl<D, P> std::ops::Neg for CochainImpl<D, P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_values(-self.values)
    }
}

impl<D, P> std::ops::Neg for &CochainImpl<D, P> {
    type Output = CochainImpl<D, P>;

    fn neg(self) -> Self::Output {
        CochainImpl::from_values(-&self.values)
    }
}

// Sub

impl<D, P> std::ops::Sub for CochainImpl<D, P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::from_values(self.values - rhs.values)
    }
}

impl<D, P> std::ops::Sub<&CochainImpl<D, P>> for CochainImpl<D, P> {
    type Output = Self;

    fn sub(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        CochainImpl::from_values(&self.values - &rhs.values)
    }
}

impl<D, P> std::ops::Sub<CochainImpl<D, P>> for &CochainImpl<D, P> {
    type Output = CochainImpl<D, P>;

    fn sub(self, rhs: CochainImpl<D, P>) -> Self::Output {
        CochainImpl::from_values(&self.values - &rhs.values)
    }
}

impl<D, P> std::ops::Sub for &CochainImpl<D, P> {
    type Output = CochainImpl<D, P>;

    fn sub(self, rhs: Self) -> Self::Output {
        CochainImpl::from_values(&self.values - &rhs.values)
    }
}

// SubAssign

impl<D, P> std::ops::SubAssign for CochainImpl<D, P> {
    fn sub_assign(&mut self, rhs: Self) {
        self.values -= rhs.values;
    }
}

impl<D, P> std::ops::SubAssign<&CochainImpl<D, P>> for CochainImpl<D, P> {
    fn sub_assign(&mut self, rhs: &CochainImpl<D, P>) {
        self.values -= &rhs.values;
    }
}

// Mul (scalar)

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
