use nalgebra as na;
use nalgebra_sparse as nas;
use typenum as tn;

use crate::{mesh::MeshPrimality, Cochain};

// The idea of this trait would be to give composability to mesh-based operators
// in a way that lets them check the dimension and primality
// of cochains they operate on at compile-time.
// Not sure if this will pan out, just sketching for now
pub trait Operator {
    type Input: OperatorInput;
    type Output: OperatorInput;

    fn apply(&self, input: &Self::Input) -> Self::Output;
    fn into_csr(self) -> nas::CsrMatrix<f64>;
}

pub trait OperatorInput {
    fn values(&self) -> &na::DVector<f64>;
    fn from_values(values: na::DVector<f64>) -> Self;
}

#[derive(Clone, Debug)]
pub struct ExteriorDerivative<Dimension, Primality> {
    // maybe this could be more efficiently implemented as a bespoke operator
    // since every row of the coboundary matrix has the same number of elements?
    // OTOH, this would make it harder to express the dual
    // (but the dual could be a separate type instead of this being generic)
    mat: nas::CsrMatrix<f64>,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
}

impl<Dimension, Primality> Operator for ExteriorDerivative<Dimension, Primality>
where
    Dimension: std::ops::Add<tn::B1>,
{
    type Input = Cochain<Dimension, Primality>;
    type Output = Cochain<tn::Add1<Dimension>, Primality>;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        Self::Output::from_values(&self.mat * input.values())
    }

    fn into_csr(self) -> nas::CsrMatrix<f64> {
        self.mat
    }
}

impl<Dimension, Primality> ExteriorDerivative<Dimension, Primality> {
    /// Constructor exposed to crate only, used in `SimplicialMesh::d`.
    pub(crate) fn new(mat: nas::CsrMatrix<f64>) -> Self {
        Self {
            mat,
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HodgeStar<Dimension, Primality> {
    // a diagonal vector is a more efficient form of storage than a CSR matrix,
    // but will make it more difficult to compose operators.
    // may need to put a CSR matrix here once the practicalities of this API are worked out
    diagonal: na::DVector<f64>,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
}

impl<Dimension, Primality> Operator for HodgeStar<Dimension, Primality>
where
    Primality: MeshPrimality,
{
    type Input = Cochain<Dimension, Primality>;
    type Output = Cochain<Dimension, Primality::Opposite>;

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

// proof of concept that composition like this can be expressed and compiles.
// ultimately this should not just hold the composed operators
// but combine them into a single matrix
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{tests::tiny_mesh_2d, Dual, Primal};

    #[test]
    fn exterior_derivative_works_in_2d() {
        let mesh = tiny_mesh_2d();
        // build a cochain where each vertex has the value of its index
        // (see mesh.rs for the mesh structure)
        let mut c0 = mesh.new_zero_cochain_primal::<tn::U0>();
        for (i, val) in c0.values.iter_mut().enumerate() {
            *val = i as f64;
        }
        let c0 = c0;

        let d0 = mesh.d::<tn::U0, Primal>();
        let c1 = d0.apply(&c0);
        // this would fail to typecheck because dimensions don't match:
        // let c2 = d0.apply(&c1);

        // this corresponds to `expected_1_simplices`
        // in `crate::mesh::tests::tiny_2d_mesh_is_correct`
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
        let c2 = mesh.d().apply(&c1);
        assert!(
            c2.values.iter().all(|v| *v == 0.0),
            "d twice should always be zero"
        );

        // for the dual d's, it suffices to check that they're
        // actually transposes of the primal ones

        assert_eq!(
            mesh.d::<tn::U0, Dual>().mat,
            mesh.d::<tn::U1, Primal>().mat.transpose(),
            "dual d_0 should be the transpose of primal d_1",
        );
        assert_eq!(
            mesh.d::<tn::U1, Dual>().mat,
            mesh.d::<tn::U0, Primal>().mat.transpose(),
            "dual d_1 should be the transpose of primal d_0",
        );
    }
}
