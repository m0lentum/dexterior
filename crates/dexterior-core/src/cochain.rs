//! Cochains, i.e. values assigned to elements of a mesh.

use nalgebra as na;

use crate::{mesh::SubsetImpl, DualCellView, SimplexView};

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
/// Thus, the more convenient alias [`Cochain`]
/// is preferred for public APIs.
#[derive(Clone)]
pub struct CochainImpl<Dimension, Primality> {
    /// The underlying vector of real values, exposed for convenience.
    ///
    /// Note that changing the dimension of this vector at runtime
    /// will cause a dimension mismatch with operators,
    /// leading to a panic when an operator is applied.
    /// Use with caution.
    ///
    /// Values can also be accessed by indexing
    /// with a corresponding [`SimplexView`] / [`DualCellView`]:
    /// ```
    /// # use dexterior_core::{mesh::tiny_mesh_3d, Cochain, Primal, Dual};
    /// # let mesh_3d = tiny_mesh_3d();
    /// let c: Cochain<1, Primal> = mesh_3d.new_zero_cochain();
    /// let c_dual: Cochain<2, Dual> = mesh_3d.star() * &c;
    ///
    /// let boundary_edges = mesh_3d.boundary::<1>();
    /// for edge in mesh_3d.simplices_in(&boundary_edges) {
    ///     let edge_val = c[edge];
    ///     let dual_val = c_dual[edge.dual()];
    ///     // ...compared to direct access
    ///     let edge_val = c.values[edge.index()];
    ///     let dual_val = c_dual.values[edge.dual().index()];
    /// }
    /// ```
    /// This way of access has the benefit of being typechecked at compile time,
    /// preventing you from accidentally indexing the wrong type of cochain.
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

    /// Replace a subset of this cochain's values with those of another cochain.
    pub fn overwrite(&mut self, subset: &SubsetImpl<Dimension, Primality>, other: &Self) {
        for i in subset.indices.ones() {
            self.values[i] = other.values[i];
        }
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

impl<'a, D, const MESH_DIM: usize> std::ops::Index<SimplexView<'a, D, MESH_DIM>>
    for CochainImpl<D, crate::Primal>
where
    D: na::DimName,
    na::Const<MESH_DIM>: na::DimNameSub<D>,
{
    type Output = f64;

    fn index(&self, simplex: SimplexView<'a, D, MESH_DIM>) -> &Self::Output {
        &self.values[simplex.index()]
    }
}

impl<'a, D, const MESH_DIM: usize> std::ops::IndexMut<SimplexView<'a, D, MESH_DIM>>
    for CochainImpl<D, crate::Primal>
where
    D: na::DimName,
    na::Const<MESH_DIM>: na::DimNameSub<D>,
{
    fn index_mut(&mut self, simplex: SimplexView<'a, D, MESH_DIM>) -> &mut Self::Output {
        &mut self.values[simplex.index()]
    }
}

impl<'a, D, const MESH_DIM: usize> std::ops::Index<DualCellView<'a, D, MESH_DIM>>
    for CochainImpl<D, crate::Dual>
where
    D: na::DimName,
    na::Const<MESH_DIM>: na::DimNameSub<D>,
{
    type Output = f64;

    fn index(&self, simplex: DualCellView<'a, D, MESH_DIM>) -> &Self::Output {
        &self.values[simplex.index()]
    }
}

impl<'a, D, const MESH_DIM: usize> std::ops::IndexMut<DualCellView<'a, D, MESH_DIM>>
    for CochainImpl<D, crate::Dual>
where
    D: na::DimName,
    na::Const<MESH_DIM>: na::DimNameSub<D>,
{
    fn index_mut(&mut self, simplex: DualCellView<'a, D, MESH_DIM>) -> &mut Self::Output {
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
