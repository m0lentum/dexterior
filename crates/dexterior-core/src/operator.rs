//! Composable operators for doing math on [`Cochain`][crate::Cochain]s.

use fixedbitset as fb;
use nalgebra as na;
use nalgebra_sparse as nas;

use crate::{cochain::CochainImpl, mesh::SubsetImpl};
use itertools::izip;

//
// traits
//

/// Trait enabling operator composition checked for compatibility at compile time.
pub trait Operator {
    /// The type of cochain this operator takes as an input.
    type Input: Operand;
    /// The type of cochain this operator produces as an output.
    type Output: Operand;

    /// Apply this operator to an input cochain.
    fn apply(&self, input: &Self::Input) -> Self::Output;
    /// Convert this operator into a CSR matrix.
    fn into_csr(self) -> nas::CsrMatrix<f64>;
}

/// Trait implemented by [`Cochain`][crate::Cochain]s to enable operators
/// to construct and deconstruct them in a generic way.
pub trait Operand {
    /// The dimension generic of this cochain, used for matching with other generic types.
    type Dimension;
    /// The primality generic of this cochain, used for matching with other generic types.
    type Primality;
    /// Get the underlying vector of values in the cochain.
    fn values(&self) -> &na::DVector<f64>;
    /// Construct a cochain from a vector of values.
    fn from_values(values: na::DVector<f64>) -> Self;
}

//
// concrete operators
//

/// A diagonal matrix operator.
///
/// See [`MatrixOperator`] and the [crate-level docs][crate#operators] for more details.
#[derive(Clone, Debug)]
pub struct DiagonalOperator<Input, Output> {
    // a diagonal vector is a more efficient form of storage than a CSR matrix.
    // this is converted to a matrix upon composition with other operators
    diagonal: na::DVector<f64>,
    _marker: std::marker::PhantomData<(Input, Output)>,
}

impl<Input, Output> Operator for DiagonalOperator<Input, Output>
where
    Input: Operand,
    Output: Operand,
{
    type Input = Input;
    type Output = Output;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        let input = input.values();
        let ret = na::DVector::from_iterator(
            input.len(),
            izip!(self.diagonal.iter(), input.iter()).map(|(&diag_val, &in_val)| diag_val * in_val),
        );
        Self::Output::from_values(ret)
    }

    fn into_csr(self) -> nas::CsrMatrix<f64> {
        // nalgebra doesn't have a method to construct CSR directly from a diagonal.
        // construct an identity matrix to get the right sparsity pattern
        // and then replace the entries
        let mut csr = nas::CsrMatrix::identity(self.diagonal.len());
        for (&diag, mat_diag) in self.diagonal.iter().zip(csr.values_mut()) {
            *mat_diag = diag;
        }
        csr
    }
}

impl<Input, Output> From<na::DVector<f64>> for DiagonalOperator<Input, Output> {
    fn from(diagonal: na::DVector<f64>) -> Self {
        Self {
            diagonal,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Input, Output> DiagonalOperator<Input, Output>
where
    Input: Operand,
    Output: Operand,
{
    /// Set a subset of elements in the output cochain to zero
    /// when this operator is applied
    /// (i.e. set a subset of rows in the operator matrix to zero).
    /// useful for boundary conditions.
    pub fn exclude_subset(
        mut self,
        set: &SubsetImpl<
            <<Self as Operator>::Output as Operand>::Dimension,
            <<Self as Operator>::Output as Operand>::Primality,
        >,
    ) -> Self {
        for row_idx in set.indices.ones() {
            self.diagonal[row_idx] = 0.0;
        }
        self
    }
}

impl<Input, Output> PartialEq for DiagonalOperator<Input, Output> {
    fn eq(&self, other: &Self) -> bool {
        self.diagonal == other.diagonal
    }
}

/// A general sparse matrix operator,
/// parameterized with the cochain types it consumes and produces.
///
/// This can be a composition of one or more [`MatrixOperator`]s and [`DiagonalOperator`]s.
/// Composition can be done using multiplication syntax:
/// ```
/// # use dexterior_core::{Primal, MatrixOperator, mesh::tiny_mesh_2d};
/// # let mesh = tiny_mesh_2d();
/// let op: MatrixOperator<_, _> = mesh.star() * mesh.d::<1, Primal>();
/// ```
/// A free function [`compose`] is also provided for the same purpose.
///
/// There is also a [`DiagonalOperator`] type for operators
/// which are specifically diagonal matrices,
/// which stores them in a somewhat more efficient format.
/// When you don't need the efficiency and prefer notational convenience,
/// these can also be converted into a MatrixOperator using the std [`From`] trait.
/// This enables writing all your operator types as `MatrixOperator<Input, Output>`,
/// a convenient pattern which can be written more concisely with the type alias [`Op`].
/// See the [crate-level docs][crate#operators] and examples for details.
#[derive(Clone, Debug)]
pub struct MatrixOperator<Input, Output> {
    mat: nas::CsrMatrix<f64>,
    _marker: std::marker::PhantomData<(Input, Output)>,
}

/// A type alias for [`MatrixOperator`]
/// to make common patterns more convenient to type.
pub type Op<Input, Output> = MatrixOperator<Input, Output>;

impl<Input, Output> Operator for MatrixOperator<Input, Output>
where
    Input: Operand,
    Output: Operand,
{
    type Input = Input;
    type Output = Output;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        Self::Output::from_values(&self.mat * input.values())
    }

    fn into_csr(self) -> nas::CsrMatrix<f64> {
        self.mat
    }
}

impl<Input, Output> MatrixOperator<Input, Output>
where
    Input: Operand,
    Output: Operand,
{
    /// Set a subset of elements in the output cochain to zero
    /// when this operator is applied
    /// (i.e. set a subset of rows in the operator matrix to zero).
    pub fn exclude_subset(
        mut self,
        set: &SubsetImpl<<Output as Operand>::Dimension, <Output as Operand>::Primality>,
    ) -> Self {
        self.mat = drop_csr_rows(self.mat, &set.indices);
        self
    }
}

impl<L, R> PartialEq for MatrixOperator<L, R> {
    fn eq(&self, other: &Self) -> bool {
        self.mat == other.mat
    }
}

// conversions from other operators and construction by matrix

impl<Input, Output> From<nas::CsrMatrix<f64>> for MatrixOperator<Input, Output> {
    fn from(mat: nas::CsrMatrix<f64>) -> Self {
        Self {
            mat,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<Input, Output> From<DiagonalOperator<Input, Output>> for MatrixOperator<Input, Output>
where
    DiagonalOperator<Input, Output>: Operator,
{
    fn from(s: DiagonalOperator<Input, Output>) -> Self {
        Self {
            mat: s.into_csr(),
            _marker: std::marker::PhantomData,
        }
    }
}

//
// helper functions
//

/// Compose two operators such that `r` is applied before `l`.
///
/// This can also be done with multiplication syntax:
/// ```
/// # use dexterior_core::{Primal, operator::compose, mesh::tiny_mesh_2d};
/// # let mesh = tiny_mesh_2d();
/// assert_eq!(
///     compose(mesh.star::<2, Primal>(), mesh.d::<1, Primal>()),
///     mesh.star::<2, Primal>() * mesh.d::<1, Primal>(),
/// );
/// ```
pub fn compose<Left, Right>(l: Left, r: Right) -> MatrixOperator<Right::Input, Left::Output>
where
    Left: Operator<Input = Right::Output>,
    Right: Operator,
{
    MatrixOperator {
        mat: l.into_csr() * r.into_csr(),
        _marker: std::marker::PhantomData,
    }
}

/// Takes a CSR matrix and generates another matrix with the given rows set to zeroes.
/// Used in operator methods for boundary conditions.
///
/// In hindsight, this could have been done much more concisely
/// by simply multiplying with a diagonal matrix.
/// Oh well, at least this one is more performant :^)
fn drop_csr_rows(mat: nas::CsrMatrix<f64>, set_to_drop: &fb::FixedBitSet) -> nas::CsrMatrix<f64> {
    let num_rows = mat.nrows();
    let num_cols = mat.ncols();
    // disassemble to reuse allocated memory
    let (mut row_offsets, mut col_indices, mut values) = mat.disassemble();

    // loop through the rows while keeping track of the number of retained values,
    // moving col_indices and values left by the appropriate amounts
    // and rebuilding row_offsets from scratch
    let mut retained_value_idx = 0;
    // row_offsets[row_idx + 1] gets overwritten during the loop,
    // so we need to keep the old value of that as state
    let mut prev_row_offset = 0;
    for row_idx in 0..num_rows {
        let old_row_range = prev_row_offset..row_offsets[row_idx + 1];
        prev_row_offset = row_offsets[row_idx + 1];

        if set_to_drop.contains(row_idx) {
            row_offsets[row_idx + 1] = row_offsets[row_idx];
        } else {
            for old_val_idx in old_row_range {
                col_indices[retained_value_idx] = col_indices[old_val_idx];
                values[retained_value_idx] = values[old_val_idx];
                retained_value_idx += 1;
            }
            row_offsets[row_idx + 1] = retained_value_idx;
        }
    }

    // drop the leftover ends off the col_indices and values
    col_indices.truncate(retained_value_idx);
    values.truncate(retained_value_idx);

    nas::CsrMatrix::try_from_csr_data(num_rows, num_cols, row_offsets, col_indices, values).unwrap()
}

//
// std trait implementations
//

// Mul implementations for composition, scalar multiplication and application to cochains.
// These need to be implemented for each type separately due to the orphan rule

// compositions

impl<In, Out, Op> std::ops::Mul<Op> for DiagonalOperator<In, Out>
where
    In: Operand,
    Out: Operand,
    Op: Operator<Output = <Self as Operator>::Input>,
{
    type Output = MatrixOperator<Op::Input, <Self as Operator>::Output>;

    fn mul(self, rhs: Op) -> Self::Output {
        compose(self, rhs)
    }
}

impl<In, Out, Op> std::ops::Mul<Op> for MatrixOperator<In, Out>
where
    In: Operand,
    Out: Operand,
    Op: Operator<Output = <Self as Operator>::Input>,
{
    type Output = MatrixOperator<Op::Input, <Self as Operator>::Output>;

    fn mul(self, rhs: Op) -> Self::Output {
        compose(self, rhs)
    }
}

// scalar multiplication

impl<Input, Output> std::ops::Mul<DiagonalOperator<Input, Output>> for f64 {
    type Output = DiagonalOperator<Input, Output>;

    fn mul(self, mut rhs: DiagonalOperator<Input, Output>) -> Self::Output {
        rhs.diagonal *= self;
        rhs
    }
}

impl<L, R> std::ops::Mul<MatrixOperator<L, R>> for f64 {
    type Output = MatrixOperator<L, R>;

    fn mul(self, mut rhs: MatrixOperator<L, R>) -> Self::Output {
        rhs.mat *= self;
        rhs
    }
}

// cochains

// impl for reference too, because the impl for value consumes the operator
// and we don't usually want that
impl<Out, D, P> std::ops::Mul<&CochainImpl<D, P>> for DiagonalOperator<CochainImpl<D, P>, Out>
where
    Out: Operand,
{
    type Output = Out;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<Out, D, P> std::ops::Mul<&CochainImpl<D, P>> for &DiagonalOperator<CochainImpl<D, P>, Out>
where
    Out: Operand,
{
    type Output = Out;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<O, D, P> std::ops::Mul<&CochainImpl<D, P>> for MatrixOperator<CochainImpl<D, P>, O>
where
    O: Operand,
{
    type Output = <Self as Operator>::Output;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<O, D, P> std::ops::Mul<&CochainImpl<D, P>> for &MatrixOperator<CochainImpl<D, P>, O>
where
    O: Operand,
{
    type Output = O;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

// impls for trait objects

impl<D, P, O> std::ops::Mul<&CochainImpl<D, P>>
    for &dyn Operator<Input = CochainImpl<D, P>, Output = O>
where
    O: Operand,
{
    type Output = O;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<D, P, O> std::ops::Mul<&CochainImpl<D, P>>
    for Box<dyn Operator<Input = CochainImpl<D, P>, Output = O>>
where
    O: Operand,
{
    type Output = O;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<D, P, O> std::ops::Mul<&CochainImpl<D, P>>
    for &Box<dyn Operator<Input = CochainImpl<D, P>, Output = O>>
where
    O: Operand,
{
    type Output = O;

    fn mul(self, rhs: &CochainImpl<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

//
// tests
//

// these are not very exhaustive;
// we'll use examples as "integration tests" to make sure the math is right
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{tiny_mesh_2d, Dual, Primal};

    #[test]
    fn exterior_derivative_works_in_2d() {
        let mesh = tiny_mesh_2d();
        // build a cochain where each vertex has the value of its index
        // (see mesh.rs for the mesh structure)
        let mut c0 = mesh.new_zero_cochain::<0, Primal>();
        for (i, val) in c0.values.iter_mut().enumerate() {
            *val = i as f64;
        }
        let c0 = c0;

        let d0 = mesh.d::<0, Primal>();
        let c1 = d0.apply(&c0);
        // this would fail to typecheck because dimensions don't match:
        // let c2 = d0.apply(&c1);

        // this corresponds to `expected_1_simplices`
        // in `crate::mesh::tests::tiny_2d_mesh_is_correct`
        #[rustfmt::skip]
        let expected_c1 = na::DVector::from_vec(vec![
            1.-0., 2.-0., 3.-0.,
            3.-1., 4.-1.,
            3.-2., 5.-2.,
            4.-3., 5.-3., 6.-3.,
            6.-4., 6.-5.,
        ]);
        assert_eq!(c1.values, expected_c1, "d_0 gave unexpected results");

        // type inference! :D
        let c2 = mesh.d() * &c1;
        assert!(
            c2.values.iter().all(|v| *v == 0.0),
            "d twice should always be zero"
        );

        // for the dual d's, it suffices to check that they're
        // actually transposes of the primal ones

        assert_eq!(
            mesh.d::<0, Dual>().mat,
            mesh.d::<1, Primal>().mat.transpose(),
            "dual d_0 should be the transpose of primal d_1",
        );
        assert_eq!(
            mesh.d::<1, Dual>().mat,
            mesh.d::<0, Primal>().mat.transpose(),
            "dual d_1 should be the transpose of primal d_0",
        );
    }

    #[test]
    fn exclude_subsets() {
        let mesh = tiny_mesh_2d();

        // d

        let d0_full = mesh.d::<0, Primal>();
        let boundary = mesh.boundary::<1>();
        let d0_excluded = d0_full.clone().exclude_subset(&boundary);
        // this would also work with type inference:
        // let d0_excluded = d0_full.clone().exclude_subset(mesh.boundary());
        // but writing out the type explicitly to make sure it's correct
        for (row_idx, (full_row, excluded_row)) in
            izip!(d0_full.mat.row_iter(), d0_excluded.mat.row_iter()).enumerate()
        {
            if boundary.indices.contains(row_idx) {
                assert!(excluded_row.nnz() == 0);
            } else {
                assert_eq!(full_row, excluded_row);
            }
        }

        // star

        let star_full = mesh.star::<2, Dual>();
        let boundary = mesh.boundary::<0>();
        let star_excluded = star_full.clone().exclude_subset(&boundary);
        for (row_idx, (full_diag, excluded_diag)) in
            izip!(star_full.diagonal.iter(), star_excluded.diagonal.iter()).enumerate()
        {
            if boundary.indices.contains(row_idx) {
                assert!(*excluded_diag == 0.0);
            } else {
                assert_eq!(full_diag, excluded_diag);
            }
        }

        // composed

        let comp_full: MatrixOperator<crate::Cochain<0, Primal>, crate::Cochain<0, Primal>> =
            mesh.star() * mesh.d() * mesh.star() * mesh.d();
        let boundary = mesh.boundary::<0>();
        let comp_excluded = comp_full.clone().exclude_subset(&boundary);
        for (row_idx, (full_row, excluded_row)) in
            izip!(comp_full.mat.row_iter(), comp_excluded.mat.row_iter()).enumerate()
        {
            if boundary.indices.contains(row_idx) {
                assert!(excluded_row.nnz() == 0);
            } else {
                assert_eq!(full_row, excluded_row);
            }
        }
    }
}
