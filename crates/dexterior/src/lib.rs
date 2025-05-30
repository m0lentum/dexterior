/*!
`dexterior` provides building blocks
for the discretization of partial differential equations
using the mathematical framework of Discrete Exterior Calculus (DEC).
These building blocks are sparse matrix operators
(exterior derivative, Hodge star) and vector operands (cochains)
associated with a simplicial mesh of any dimension.
An effort is made to provide as many compile-time checks
and type inference opportunities as possible
to produce concise and correct code.

Additionally, `dexterior` provides some batteries-included tools
for real-time visualization with [wgpu] in the optional `visuals` module.
Enable the `visuals` feature to use them.
These are less mature than the core library,
currently only supporting 2D use cases.

# Example

To give you a taste of the basic features available,
the following code implements a simple acoustic wave simulation:

```
use dexterior as dex;
use nalgebra::Vector2;

// type aliases for simulation variables ease readability
type Pressure = dex::Cochain<0, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;

// meshes can be loaded from files generated with `gmsh`
let msh_bytes = include_bytes!("../examples/meshes/2d_square_pi_x_pi.msh");
let mesh = dex::gmsh::load_trimesh_2d(msh_bytes).expect("Failed to load mesh");

let dt = 0.1;
// generic operator expressed with its input and output type.
// type inference figures out which dimensions of `star` and `d` we need
// to make this expression match the type
let p_step: dex::Op<Velocity, Pressure> =
    -dt * mesh.star() * mesh.d() * mesh.star();
// type inference also knows which `d` we mean here
// because we multiply `p` by this operator later
let v_step = dt * mesh.d();

// integrate an initial state function into discrete cochains
let mut p: Pressure = mesh.integrate_cochain(dex::quadrature::Pointwise(
    |v: Vector2<f64>| f64::sin(3.0 * v.x) * f64::sin(2.0 * v.y)
));
let mut v: Velocity = mesh.new_zero_cochain();

// step the simulation forward in time
for _step in 0..10 {
    p += &p_step * &v;
    v += &v_step * &p;
}
```

For examples including visuals, see the examples in the [repo].

# Meshes

The core of a DEC discretization is the computation mesh,
represented in the [`SimplicialMesh`] type.
The type has one generic parameter representing the dimension of its embedding space
and consequently the dimension of its highest-dimensional set of simplices.

## Constructing a mesh

A mesh can be loaded from a `.msh` file generated with [gmsh](http://gmsh.info/).
2D triangle meshes and 3D tetrahedral meshes are supported.
See the [`gmsh`] module for details.

Alternatively, a mesh can be built manually from a list of vertices
and a list of indices defining the highest-dimensional simplices.

# Working with cochains and operators

Simulation variables in DEC-based simulations are expressed as cochains,
which associate a value with every cell of a specific dimension
in either the primal or the dual mesh.
In this library, both the dimension and the association with the primal or dual mesh
are expressed via generic parameters on [`SimplicialMesh`] methods.
Dimension is expressed as a `usize` constant and the type of associated mesh
as either [`Primal`] or [`Dual`].

Say we have a two-dimensional mesh,
```
# use dexterior::{SimplicialMesh, mesh::tiny_mesh_2d as some_mesh};
let mesh: SimplicialMesh<2> = some_mesh();
```
The following code constructs two cochains,
the first associated with primal 1-simplices (i.e. edges of triangles)
and the second with dual 0-simplices (i.e. vertices at the circumcenters of triangles),
on this mesh, with zeroes for all elements:
```
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
# let mesh = tiny_mesh_2d();
let primal_1_cochain = mesh.new_zero_cochain::<1, Primal>();
let dual_0_cochain = mesh.new_zero_cochain::<0, Dual>();
```
To construct cochains with nonzero values
by integrating a function over mesh cells,
see [`SimplicialMesh::integrate_cochain`]
and the [`quadrature`] module.

## Operators

The two most important operators in DEC are the exterior derivative and Hodge star,
which are constructed from a [`SimplicialMesh`].
They use generics similar to cochains to ensure compatible inputs and outputs.

### Exterior derivative

The `k`-dimensional exterior derivative takes a `k`-cochain
on either the primal or dual mesh
and produces a `k+1`-cochain on the same mesh.
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
# let mesh = tiny_mesh_2d();
# let primal_d_1 = mesh.d::<1, Primal>();
# let primal_1_cochain = mesh.new_zero_cochain::<1, Primal>();
let result: Cochain<2, Primal> = &primal_d_1 * &primal_1_cochain;
```

### Hodge star

The Hodge star maps cochains from the primal mesh to the dual and vice versa.
Hodge star operators are constructed with [`SimplicialMesh::star`][crate::SimplicialMesh::star];
for example,
```
# use dexterior::{mesh::tiny_mesh_2d, Primal};
# let mesh = tiny_mesh_2d();
let primal_star_1 = mesh.star::<1, Primal>();
```
constructs a Hodge star mapping primal 1-cochains to dual `n-1`-cochains,
where `n` is the dimension of the mesh.
The inverse of the `k`-dimensional star is the dual `n-k`-dimensional star,
with a sign depending on the dimensions of the mesh and the star.
Applying both in succession is equivalent to multiplication by `(-1)^(k * (n - k))`.
Let us illustrate this with a 3D mesh:
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
# use dexterior::{mesh::tiny_mesh_3d, Primal, Dual, MatrixOperator};
# let mesh = tiny_mesh_3d();
let divergence: MatrixOperator<_, _>
    = mesh.star::<3, Dual>()
    * mesh.d::<2, Dual>()
    * mesh.star::<1, Primal>();
```
The resulting type is an operator that takes the rightmost operator's input
and produces the leftmost operator's output:
```
# use dexterior::{mesh::tiny_mesh_3d, Primal, Dual, Cochain};
# let mesh = tiny_mesh_3d();
# let divergence = mesh.star() * mesh.d() * mesh.star();
let input: Cochain<1, Primal> = mesh.new_zero_cochain();
let output: Cochain<0, Primal> = &divergence * &input;
```

This may seem verbose, and it is, but don't worry --
as we'll see in the next section,
most of these type annotations don't actually need to be written.

### Compile-time checks and type inference

When composing operators or applying them to cochains,
the compatibility of the operands is checked at compile time.
This is achieved via the [`Operator`] trait,
which tracks the input and output cochain types of an operator.
For example, all of the following would fail to compile:
```compile_fail
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
# let mesh = tiny_mesh_2d();
mesh.d::<1, Primal>() * &mesh.new_zero_cochain::<2, Primal>();
```
```compile_fail
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
# let mesh = tiny_mesh_2d();
mesh.star::<2, Dual>() * &mesh.new_zero_cochain::<2, Primal>();
```
```compile_fail
# use dexterior::{mesh::tiny_mesh_2d, Primal, Dual};
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
or like this, annotating the resulting [`MatrixOperator`]'s
input and output types instead:
```
# use dexterior::{mesh::tiny_mesh_3d, Cochain, Primal, MatrixOperator};
# let mesh = tiny_mesh_3d();
let divergence: MatrixOperator<Cochain<1, Primal>, Cochain<0, Primal>>
    = mesh.star() * mesh.d() * mesh.star();
```
A common pattern is to give [`Cochain`]s named type aliases
based on the simulation variables they represent,
which enables nice definitions like this one from the `membrane` example
(using the [`Op`] type alias for brevity):
```
# use dexterior::{mesh::tiny_mesh_2d, Cochain, Op, Primal};
# let mesh = tiny_mesh_2d();
type Pressure = Cochain<0, Primal>;
type Velocity = Cochain<1, Primal>;

struct Ops {
    p_step: Op<Velocity, Pressure>,
    v_step: Op<Pressure, Velocity>,
}

let ops = Ops {
    p_step: mesh.star() * mesh.d() * mesh.star(),
    v_step: mesh.d().into(),
};
```

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

### Visualization

Real-time visualization of the computation mesh and values of cochains
is available in the `dexterior-visuals` crate.
Currently only a few specific cases are supported; more to come later.
See the examples in the [repo] for usage.

[repo]: https://github.com/m0lentum/dexterior
[pydec]: https://github.com/hirani/pydec/
[wgpu]: https://wgpu.rs
*/

#[doc(inline)]
pub use dexterior_core::*;

#[cfg(feature = "visuals")]
pub use dexterior_visuals as visuals;
