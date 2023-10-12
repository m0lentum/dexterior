/*! NOTE: this library is a work in progress and not ready for use yet.

Dexterior provides building blocks
for the discretization of partial differential equations
using Discrete Exterior Calculus (DEC).
A simplicial mesh of any dimension can be processed into a DEC primal/dual complex pair,
on which cochains and operators can be defined.
To the extent possible, the existence and interactions of operators and cochains
is checked at compile-time to ensure correct use.

# Recommended reading

This documentation will not explain what exterior calculus is
or how to derive equations to solve with this library.
For curious readers unfamiliar with the topic or in need of a refresher,
[the author's master's thesis](http://urn.fi/URN:NBN:fi:jyu-202310035379)
provides a relatively accessible tutorial.
For more detail and mathematical rigor (but still an approachable tone),
see the following texts:
- Blair Perot, J. & Zusi, C. (2014). [Differential forms for scientists and engineers
  ](https://www.sciencedirect.com/science/article/pii/S0021999113005354)
- Crane, K. et al. (2013). [Digital geometry processing with discrete exterior calculus
  ](https://dl.acm.org/doi/pdf/10.1145/2504435.2504442)
- Desbrun, M., Kanso, E. & Tong, Y. (2006). [Discrete differential forms
  for Computational Modeling](https://dl.acm.org/doi/pdf/10.1145/1185657.1185665)

# Constructing a complex

The core of a DEC discretization is the computation mesh,
represented in the [`SimplicialMesh`][crate::SimplicialMesh] type.
The type has one generic parameter representing the dimension of its embedding space
and consequently the dimension of its highest-dimensional set of simplices.
Currently (as this is a heavy work in progress),
you can only create one by supplying a list of raw vertices and indices.
The plan is to support meshes built with [gmsh](http://gmsh.info/)
and maybe eventually include a purpose-built mesh generator in Dexterior itself.

## The dual complex

TODO: explain how the dual relates to the primal

# Working with cochains and operators

Simulation variables in DEC-based simulations are expressed as cochains,
which associate a value with every cell of a specific dimension
in either the primal or the dual complex.
In this library, both the dimension and the association with the primal or dual complex
are expressed via generic parameters on [`SimplicialMesh`][crate::SimplicialMesh] methods.
Dimension is expressed as a `usize` constant and the type of associated complex
as either [`Primal`][crate::Primal] or [`Dual`][crate::Dual].

Say we have a two-dimensional mesh,
```
# use dexterior::{SimplicialMesh, mesh::tiny_mesh_2d as some_mesh};
let mesh: SimplicialMesh<2> = some_mesh();
```
The following code constructs two cochains,
the first associated with primal 1-simplices
and the second with dual 0-simplices (i.e. vertices), on this mesh,
with zeroes for all elements:
```
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
# let mesh = tiny_mesh_2d();
let primal_1_cochain = mesh.new_zero_cochain::<1, Primal>();
let dual_0_cochain = mesh.new_zero_cochain::<0, Dual>();
```

## Operators

The two most important operators in DEC are the exterior derivative and Hodge star,
which are constructed from a [`SimplicialMesh`][crate::SimplicialMesh].
They use generics similar to cochains to ensure compatible inputs and outputs.

### Exterior derivative

The `k`-dimensional exterior derivative takes a `k`-cochain
on either the primal or dual complex
and produces a `k+1`-cochain on the same complex.
Exterior derivative operators are constructed with [`SimplicialMesh::d`][crate::SimplicialMesh::d];
for example,
```
# use dexterior::{mesh::tiny_mesh_2d, Primal};
# let mesh = tiny_mesh_2d();
let primal_d_1 = mesh.d::<1, Primal>();
```
constructs a primal `1`-dimensional exterior derivative.
To apply it to a primal 1-cochain,
we can use multiplication syntax:
```
# use dexterior::{mesh::tiny_mesh_2d, Primal, Cochain};
# use nalgebra::Const;
# let mesh = tiny_mesh_2d();
# let primal_d_1 = mesh.d::<1, Primal>();
# let primal_1_cochain = mesh.new_zero_cochain::<1, Primal>();
let result: Cochain<Const<2>, Primal> = primal_d_1 * &primal_1_cochain;
```

(Note: for technical reasons, the [`Cochain`][crate::Cochain] types's generics
use `nalgebra`'s [`Const`][nalgebra::Const] instead of `const usize`.)

### Hodge star

The Hodge star maps cochains from the primal complex to the dual and vice versa.
Hodge star operators are constructed with [`SimplicialMesh::star`][crate::SimplicialMesh::star];
for example,
```
# use dexterior::{mesh::tiny_mesh_2d, Primal};
# let mesh = tiny_mesh_2d();
let primal_star_1 = mesh.star::<1, Primal>();
```
constructs a Hodge star mapping primal 1-cochains to dual `n-1`-cochains,
where `n` is the dimension of the mesh.
Its inverse is the dual `n-1`-dimensional star,
and applying both in succession is equivalent to the identity operation.
Let us illustrate this with a 3D mesh to make the dimensions more clear:
```
# use dexterior::{mesh::tiny_mesh_3d as some_mesh, SimplicialMesh, Primal, Dual};
let mesh: SimplicialMesh<3> = some_mesh();
let primal_1_cochain = mesh.new_zero_cochain::<1, Primal>();
assert_eq!(
    mesh.star::<2, Dual>() * mesh.star::<1, Primal>() * &primal_1_cochain,
    primal_1_cochain,
);
```

### Operator composition

Most exterior calculus equations feature compositions of multiple operators,
e.g. the 3-dimensional divergence `∇· = ٭d٭`.
Rather than operating on a cochain multiple times in succession with all these operators,
it is desirable to first compose the operators into a single matrix.
This can be done with multiplication syntax:
```
# use dexterior::{mesh::tiny_mesh_3d, Primal, Dual, ComposedOperator};
# let mesh = tiny_mesh_3d();
let divergence: ComposedOperator<_, _>
    = mesh.star::<3, Dual>()
    * mesh.d::<2, Dual>()
    * mesh.star::<1, Primal>();
```
The resulting type is an operator that takes the rightmost operator's input
and produces the leftmost operator's output:
```
# use dexterior::{mesh::tiny_mesh_3d, Primal, Dual, Cochain};
# use nalgebra::Const;
# let mesh = tiny_mesh_3d();
# let divergence = mesh.star() * mesh.d() * mesh.star();
let input: Cochain<Const<1>, Primal> = mesh.new_zero_cochain();
let output: Cochain<Const<0>, Primal> = divergence * &input;
```

This may seem verbose, and it is, but don't worry --
as we'll see in the next section,
most of these type annotations don't actually need to be written.

### Compile-time checks and type inference

When composing operators or applying them to cochains,
the compatibility of the operands is checked at compile time.
This is achieved via the [`Operator`][crate::operator::Operator] trait,
which tracks the input and output cochain types of an operator.
For example, all of the following would fail to compile:
```compile_fail
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
# use nalgebra::Const;
# let mesh = tiny_mesh_2d();
mesh.d::<1, Primal>() * &mesh.new_zero_cochain::<2, Primal>();
```
```compile_fail
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
# use nalgebra::Const;
# let mesh = tiny_mesh_2d();
mesh.star::<2, Dual>() * &mesh.new_zero_cochain::<2, Primal>();
```
```compile_fail
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
# use nalgebra::Const;
# let mesh = tiny_mesh_2d();
mesh.star::<0, Dual>() * &mesh.d::<1, Primal>();
```

Thanks to these constraints, the compiler can infer the types of most generic expressions.
The earlier divergence example could be written like this:
```
# use dexterior::{mesh::tiny_mesh_3d, Primal};
# let mesh = tiny_mesh_3d();
let divergence = mesh.star() * mesh.d() * mesh.star::<1, Primal>();
```
In general, it's enough to give the type of one operator or cochain
and the rest will follow.
If you apply the divergence to a cochain in the same function,
```
# use dexterior::{mesh::tiny_mesh_3d, Primal};
# let mesh = tiny_mesh_3d();
let divergence = mesh.star() * mesh.d() * mesh.star();
let input = mesh.new_zero_cochain::<1, Primal>();
let output = divergence * &input;
```
also compiles.

In addition to the compatibility of operands,
the existence of cochains and operators is also ensured at compile time.
This means you cannot construct cochains or Hodge stars of higher dimension than your mesh
or exterior derivatives whose output dimension would be higher than the mesh's
(i.e. the maximum dimension for `d` is `n-1`).
For a 2D mesh, none of the following would compile:
```compile_fail
# use dexterior::{Primal, mesh::tiny_mesh_2d};
# let mesh = tiny_mesh_2d();
let c = mesh.new_zero_cochain::<3, Primal>();
```
```compile_fail
# use dexterior::{Primal, mesh::tiny_mesh_2d};
# let mesh = tiny_mesh_2d();
let star = mesh.star::<3, Primal>();
```
```compile_fail
# use dexterior::{Primal, mesh::tiny_mesh_2d};
# let mesh = tiny_mesh_2d();
let d = mesh.d::<2, Primal>();
```

### Pain points and workarounds

Unfortunately, the type errors arising from mismatching operands
are rather verbose and confusing,
complaining about missing trait implementations.
A good rule of thumb is to double check your operators
when such an error presents itself.

A more problematic downside to this approach to operator types
is that composition results in extremely verbose types.
The type of the divergence operator, for instance, is
```
# use dexterior::{mesh::tiny_mesh_3d, Primal, Dual, ComposedOperator, HodgeStar, ExteriorDerivative};
# let mesh = tiny_mesh_3d();
let divergence: ComposedOperator<
    ComposedOperator<HodgeStar<3, 3, Dual>, ExteriorDerivative<2, Dual>>,
    HodgeStar<1, 3, Primal>,
>
    = mesh.star() * mesh.d() * mesh.star();
```
If you need to store an operator in a struct,
an alternative to writing all this out is to store it as a trait object:
```
# use dexterior::{mesh::tiny_mesh_3d, Primal, Operator, Cochain};
# use nalgebra::Const;
# let mesh = tiny_mesh_3d();
struct MyOperators {
    divergence: Box<dyn Operator<
        Input = Cochain<Const<1>, Primal>,
        Output = Cochain<Const<0>, Primal>,
    >>,
}
let ops = MyOperators {
    divergence: Box::new(mesh.star() * mesh.d() * mesh.star()),
};

// TODO: actually this doesn't work right now
// because Mul is not implemented for `dyn Operator`.
// This pattern should be supported.

let input = mesh.new_zero_cochain::<1, Primal>();
let output = *ops.divergence * &input;
```
(another TODO: can we get the Cochain type take a const generic instead of
`nalgebra::Const`? That would improve the ergonomics of this pattern quite a bit.
Alternatively, type aliases may be helpful here.)

Alternatively, you can opt out of `dexterior`'s type checks
by converting your operators into [`nalgebra_sparse::CsrMatrix`][nalgebra_sparse::CsrMatrix],
which can be multiplied with the [`nalgebra::DVector`][nalgebra::DVector]
stored in the `values` field of [`Cochain`][crate::Cochain]:
```
# use dexterior::{mesh::tiny_mesh_3d, Primal, Cochain};
# use nalgebra_sparse::CsrMatrix;
# use nalgebra::Const;
# let mesh = tiny_mesh_3d();
struct MyOperators {
    divergence: CsrMatrix<f64>,
}
// the `into_csr` method is from the Operator trait, so it needs to be in scope
use dexterior::Operator;
let divergence = mesh.star() * mesh.d() * mesh.star::<1, Primal>();
let ops = MyOperators {
    divergence: divergence.into_csr(),
};

let input: nalgebra::DVector<f64> = mesh.new_zero_cochain::<1, Primal>().values;
let output: nalgebra::DVector<f64> = ops.divergence * &input;
```

*/

#![warn(missing_docs)]

pub mod mesh;
#[doc(inline)]
pub use mesh::{Dual, Primal, SimplexIter, SimplexView, SimplexViewMut, SimplicialMesh};

pub mod cochain;
#[doc(inline)]
pub use cochain::Cochain;

pub mod operator;
#[doc(inline)]
pub use operator::{ComposedOperator, ExteriorDerivative, HodgeStar, Operator};
