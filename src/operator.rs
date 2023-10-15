//! Composable operators for doing math on [`Cochain`][crate::Cochain]s.

use nalgebra as na;
use nalgebra_sparse as nas;

use crate::{simplicial_complex::MeshPrimality, Cochain};

/// Trait enabling operator composition checked for compatibility at compile time.
pub trait Operator {
    /// The type of cochain this operator takes as an input.
    type Input: OperatorInput;
    /// The type of cochain this operator produces as an output.
    type Output: OperatorInput;

    /// Method to apply this operator to an input cochain.
    fn apply(&self, input: &Self::Input) -> Self::Output;
    /// Method to convert this operator into a CSR matrix for composition.
    fn into_csr(self) -> nas::CsrMatrix<f64>;
}

/// Trait implemented by [`Cochain`]s to enable operators
/// to construct and deconstruct them in a generic way.
pub trait OperatorInput {
    /// Get the underlying vector of values in the cochain.
    fn values(&self) -> &na::DVector<f64>;
    /// Construct a cochain from a vector of values.
    fn from_values(values: na::DVector<f64>) -> Self;
}

/// Convenience type alias for concisely expressing operators as trait objects.
///
/// The first two generics are for the input cochain
/// and the last two for the output.
/// See the [module-level docs][crate#pain-points-and-workarounds] for a usage example.
pub type DynOp<const IN_DIM: usize, InPrimality, const OUT_DIM: usize, OutPrimality> = dyn Operator<
    Input = Cochain<na::Const<IN_DIM>, InPrimality>,
    Output = Cochain<na::Const<OUT_DIM>, OutPrimality>,
>;

/// The exterior derivative, also known as the coboundary operator.
///
/// This operator is constructed from a mesh with [`SimplicialComplex::d`][crate::SimplicialComplex::d].
/// See its documentation for details.
#[derive(Clone, Debug)]
pub struct ExteriorDerivative<const DIM: usize, Primality> {
    mat: nas::CsrMatrix<f64>,
    _marker: std::marker::PhantomData<Primality>,
}

impl<const DIM: usize, Primality> Operator for ExteriorDerivative<DIM, Primality>
where
    na::Const<DIM>: na::DimNameAdd<na::U1>,
{
    type Input = Cochain<na::Const<DIM>, Primality>;
    type Output = Cochain<na::DimNameSum<na::Const<DIM>, na::U1>, Primality>;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        Self::Output::from_values(&self.mat * input.values())
    }

    fn into_csr(self) -> nas::CsrMatrix<f64> {
        self.mat
    }
}

impl<const DIM: usize, Primality> ExteriorDerivative<DIM, Primality> {
    /// Constructor exposed to crate only, used in `SimplicialComplex::d`.
    pub(crate) fn new(mat: nas::CsrMatrix<f64>) -> Self {
        Self {
            mat,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<const DIM: usize, Primality> PartialEq for ExteriorDerivative<DIM, Primality> {
    fn eq(&self, other: &Self) -> bool {
        self.mat == other.mat
    }
}

/// A diagonal Hodge star operator.
///
/// This operator is constructed from a mesh with [`SimplicialComplex::star`][crate::SimplicialComplex::star].
/// See its documentation for details.
#[derive(Clone, Debug)]
pub struct HodgeStar<const DIM: usize, const MESH_DIM: usize, Primality> {
    // a diagonal vector is a more efficient form of storage than a CSR matrix.
    // this is converted to a matrix upon composition with other operators
    diagonal: na::DVector<f64>,
    _marker: std::marker::PhantomData<Primality>,
}

impl<const DIM: usize, const MESH_DIM: usize, Primality> Operator
    for HodgeStar<DIM, MESH_DIM, Primality>
where
    na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    Primality: MeshPrimality,
{
    type Input = Cochain<na::Const<DIM>, Primality>;
    type Output =
        Cochain<na::DimNameDiff<na::Const<MESH_DIM>, na::Const<DIM>>, Primality::Opposite>;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        let input = input.values();
        let mut ret = na::DVector::zeros(input.len());
        for ((&in_val, &diag_val), ret_val) in
            input.iter().zip(self.diagonal.iter()).zip(ret.iter_mut())
        {
            *ret_val = in_val * diag_val;
        }
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

impl<const DIM: usize, const MESH_DIM: usize, Primality> HodgeStar<DIM, MESH_DIM, Primality> {
    /// Constructor exposed to crate only, used in `SimplicialComplex::star`.
    pub(crate) fn new(diagonal: na::DVector<f64>) -> Self {
        Self {
            diagonal,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<const DIM: usize, const MESH_DIM: usize, Primality> PartialEq
    for HodgeStar<DIM, MESH_DIM, Primality>
{
    fn eq(&self, other: &Self) -> bool {
        self.diagonal == other.diagonal
    }
}

/// A composition of two [`Operator`][self::Operator]s.
///
/// Operator composition can be done using multiplication syntax:
/// ```
/// # use dexterior::{Primal, ComposedOperator, simplicial_complex::tiny_mesh_2d};
/// # let mesh = tiny_mesh_2d();
/// let op: ComposedOperator<_, _> = mesh.star() * mesh.d::<1, Primal>();
/// ```
/// A free function [`compose`][self::compose] is also provided for the same purpose.
#[derive(Clone, Debug)]
pub struct ComposedOperator<Left, Right> {
    mat: nas::CsrMatrix<f64>,
    _marker: std::marker::PhantomData<(Left, Right)>,
}

impl<Left, Right> Operator for ComposedOperator<Left, Right>
where
    Left: Operator<Input = Right::Output>,
    Right: Operator,
{
    type Input = Right::Input;
    type Output = Left::Output;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        Self::Output::from_values(&self.mat * input.values())
    }

    fn into_csr(self) -> nas::CsrMatrix<f64> {
        self.mat
    }
}

impl<L, R> PartialEq for ComposedOperator<L, R> {
    fn eq(&self, other: &Self) -> bool {
        self.mat == other.mat
    }
}

/// Compose two operators such that `r` is applied before `l`.
///
/// This can also be done with multiplication syntax:
/// ```
/// # use dexterior::{Primal, operator::compose, simplicial_complex::tiny_mesh_2d};
/// # let mesh = tiny_mesh_2d();
/// assert_eq!(
///     compose(mesh.star::<2, Primal>(), mesh.d::<1, Primal>()),
///     mesh.star::<2, Primal>() * mesh.d::<1, Primal>(),
/// );
/// ```
pub fn compose<Left, Right>(l: Left, r: Right) -> ComposedOperator<Left, Right>
where
    Left: Operator<Input = Right::Output>,
    Right: Operator,
{
    ComposedOperator {
        mat: l.into_csr() * r.into_csr(),
        _marker: std::marker::PhantomData,
    }
}

// Mul implementations for composition and application to cochains.
// These need to be implemented for each type separately due to the orphan rule

// compositions

impl<const D: usize, P, Op> std::ops::Mul<Op> for ExteriorDerivative<D, P>
where
    na::Const<D>: na::DimNameAdd<na::U1>,
    Op: Operator<Output = <Self as Operator>::Input>,
{
    type Output = ComposedOperator<Self, Op>;

    fn mul(self, rhs: Op) -> Self::Output {
        compose(self, rhs)
    }
}

impl<const D: usize, const MD: usize, P, Op> std::ops::Mul<Op> for HodgeStar<D, MD, P>
where
    na::Const<MD>: na::DimNameSub<na::Const<D>>,
    P: MeshPrimality,
    Op: Operator<Output = <Self as Operator>::Input>,
{
    type Output = ComposedOperator<Self, Op>;

    fn mul(self, rhs: Op) -> Self::Output {
        compose(self, rhs)
    }
}

impl<L, R, Op> std::ops::Mul<Op> for ComposedOperator<L, R>
where
    L: Operator<Input = R::Output>,
    R: Operator,
    Op: Operator<Output = <Self as Operator>::Input>,
{
    type Output = ComposedOperator<Self, Op>;

    fn mul(self, rhs: Op) -> Self::Output {
        compose(self, rhs)
    }
}

// cochains

impl<const D: usize, P> std::ops::Mul<&Cochain<na::Const<D>, P>> for ExteriorDerivative<D, P>
where
    na::Const<D>: na::DimNameAdd<na::U1>,
{
    type Output = <Self as Operator>::Output;

    fn mul(self, rhs: &Cochain<na::Const<D>, P>) -> Self::Output {
        self.apply(rhs)
    }
}

// impl for reference too, because the impl for value consumes the operator
// and we don't usually want that
impl<const D: usize, P> std::ops::Mul<&Cochain<na::Const<D>, P>> for &ExteriorDerivative<D, P>
where
    na::Const<D>: na::DimNameAdd<na::U1>,
{
    type Output = <ExteriorDerivative<D, P> as Operator>::Output;

    fn mul(self, rhs: &Cochain<na::Const<D>, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<const D: usize, const MD: usize, P> std::ops::Mul<&Cochain<na::Const<D>, P>>
    for HodgeStar<D, MD, P>
where
    na::Const<MD>: na::DimNameSub<na::Const<D>>,
    P: MeshPrimality,
{
    type Output = <Self as Operator>::Output;

    fn mul(self, rhs: &Cochain<na::Const<D>, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<const D: usize, const MD: usize, P> std::ops::Mul<&Cochain<na::Const<D>, P>>
    for &HodgeStar<D, MD, P>
where
    na::Const<MD>: na::DimNameSub<na::Const<D>>,
    P: MeshPrimality,
{
    type Output = <HodgeStar<D, MD, P> as Operator>::Output;

    fn mul(self, rhs: &Cochain<na::Const<D>, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<L, R, D, P> std::ops::Mul<&Cochain<D, P>> for ComposedOperator<L, R>
where
    L: Operator<Input = R::Output>,
    R: Operator<Input = Cochain<D, P>>,
{
    type Output = <Self as Operator>::Output;

    fn mul(self, rhs: &R::Input) -> Self::Output {
        self.apply(rhs)
    }
}

impl<L, R, D, P> std::ops::Mul<&Cochain<D, P>> for &ComposedOperator<L, R>
where
    L: Operator<Input = R::Output>,
    R: Operator<Input = Cochain<D, P>>,
{
    type Output = <ComposedOperator<L, R> as Operator>::Output;

    fn mul(self, rhs: &R::Input) -> Self::Output {
        self.apply(rhs)
    }
}

// impls for trait objects

impl<D, P, O> std::ops::Mul<&Cochain<D, P>> for &dyn Operator<Input = Cochain<D, P>, Output = O>
where
    O: OperatorInput,
{
    type Output = O;

    fn mul(self, rhs: &Cochain<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<D, P, O> std::ops::Mul<&Cochain<D, P>> for Box<dyn Operator<Input = Cochain<D, P>, Output = O>>
where
    O: OperatorInput,
{
    type Output = O;

    fn mul(self, rhs: &Cochain<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

impl<D, P, O> std::ops::Mul<&Cochain<D, P>>
    for &Box<dyn Operator<Input = Cochain<D, P>, Output = O>>
where
    O: OperatorInput,
{
    type Output = O;

    fn mul(self, rhs: &Cochain<D, P>) -> Self::Output {
        self.apply(rhs)
    }
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplicial_complex::{tiny_mesh_2d, Dual, Primal};

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
        // in `crate::simplicial_complex::tests::tiny_2d_mesh_is_correct`
        #[rustfmt::skip]
        let expected_c1 = na::DVector::from_vec(vec![
            3.0 - 2.0, 3.0 - 0.0, 2.0 - 0.0,
            3.0 - 1.0, 1.0 - 0.0,
            4.0 - 3.0, 4.0 - 1.0,
            5.0 - 3.0, 5.0 - 2.0,
            6.0 - 5.0, 6.0 - 3.0,
            6.0 - 4.0,
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

    /// Operators can be chained (and have their types inferred correctly).
    /// This one is mostly about types and syntax,
    /// so this compiling is a success in itself.
    ///
    /// TODO: move this into a doctest somewhere
    #[test]
    fn operator_composition_works() {
        let mesh = tiny_mesh_2d();

        let c1 = mesh.new_zero_cochain::<1, Primal>();
        let complicated_op =
            mesh.star() * mesh.d() * mesh.d() * mesh.star() * mesh.d() * mesh.star();
        let _res: Cochain<na::Const<0>, Dual> = complicated_op * &c1;
    }
}
