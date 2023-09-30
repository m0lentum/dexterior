use nalgebra as na;
use nalgebra_sparse as nas;
use typenum as tn;

use crate::cochain::Cochain;

// The idea of this trait would be to give composability to mesh-based operators
// in a way that lets them check the dimension and primality
// of cochains they operate on at compile-time.
// Not sure if this will pan out, just sketching for now
pub trait Operator {
    type Input;
    type Output;

    fn apply(&self, input: &Self::Input) -> Self::Output;
}

pub struct ExteriorDerivative<Dimension, Primality> {
    // maybe this could be more efficiently implemented as a bespoke operator
    // since every row of the coboundary matrix has the same number of elements?
    // OTOH, this would make composition harder
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
        todo!()
    }
}

pub struct HodgeStar<Dimension, Primality> {
    // a diagonal vector is a more efficient form of storage than a CSR matrix,
    // but will make it more difficult to compose operators.
    // may need to put a CSR matrix here once the practicalities of this API are worked out
    diagonal: na::DVector<f64>,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
}

impl<Dimension, Primality> Operator for HodgeStar<Dimension, Primality>
where
    Primality: crate::cochain::MeshPrimality,
{
    type Input = Cochain<Dimension, Primality>;
    type Output = Cochain<Dimension, Primality::Opposite>;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        todo!()
    }
}

// proof of concept that composition like this can be expressed and compiles.
// ultimately this should not just hold the composed operators
// but combine them into a single matrix
pub struct ComposedOperator<Left, Right>(Left, Right);

impl<Left, Right> Operator for ComposedOperator<Left, Right>
where
    Left: Operator<Input = Right::Output>,
    Right: Operator,
{
    type Input = Right::Input;
    type Output = Left::Output;

    fn apply(&self, input: &Self::Input) -> Self::Output {
        let Self(left, right) = self;
        left.apply(&right.apply(input))
    }
}
