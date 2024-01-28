# dexterior

`dexterior` provides building blocks
for the discretization of partial differential equations
using the mathematical framework of Discrete Exterior Calculus (DEC).
These building blocks are sparse matrix operators
(exterior derivative, Hodge star) and vector operands (cochains)
associated with a simplicial mesh of any dimension.
An effort is made to provide as many compile-time checks
and type inference opportunities as possible
to produce concise and correct code.

A companion crate, `dexterior-visuals`, provides some batteries-included tools
to visualize simulations implemented with `dexterior` in real time.

## Current state

This library is being built to facilitate the author's PhD studies
and as such is developed with their use cases as first priority.
The main library is already fairly far along in development,
implementing most of [PyDEC]'s functionality relevant to PDEs,
but it hasn't been tested with many cases yet.
Some known rough edges remain.
The visuals crate is quite limited at the moment,
supporting only a few specific 2D use cases.

## Example

To give you a taste of the basic features available,
the following code implements a simple acoustic wave simulation:

```rust
use dexterior as dex;

// type aliases for simulation variables ease readability
type Pressure = dex::Cochain<0, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;

// meshes can be loaded from files generated with `gmsh`
let msh_bytes = include_bytes!("./examples/meshes/2d_square_pi_x_pi.msh");
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
let mut p: Pressure = mesh.integrate_cochain(|v| {
    f64::sin(3.0 * v[0].x) * f64::sin(2.0 * v[0].y)
});
let mut v: Velocity = mesh.new_zero_cochain();

// step the simulation forward in time
for _step in 0..10 {
    p += &p_step * &v;
    v += &v_step * &p;
}
```

## References and recommended reading

`dexterior` is largely inspired by the [PyDEC] library for Python,
the most comprehensive implementation of DEC I know of.
The accompanying paper [PyDEC: Software and Algorithms
for Discretization of Exterior Calculus](https://dl.acm.org/doi/10.1145/2382585.2382588)
by Hirani and Bell (2012) has been an invaluable aid in development.

More articles I wish to credit:
- Desbrun, M., Hirani, A., Leok, M. & Marsden, J. (2005).
  [Discrete Exterior Calculus](https://arxiv.org/pdf/math/0508341.pdf)
- Hirani, A., Kalyanaraman, K. & VanderZee, E. (2012).
  [Delaunay Hodge star](https://www.sciencedirect.com/science/article/pii/S0010448512002436)

For a tutorial on the basics of DEC and in-depth explanations
of the acoustics examples in this repo, see my master's thesis:
- Myyrä, M. (2023). [Discrete exterior calculus and exact controllability
  for time-harmonic acoustic wave simulation](http://urn.fi/URN:NBN:fi:jyu-202310035379)

Additional recommendations for readers new to the topic or in need of a refresher:
- Desbrun, M., Kanso, E. & Tong, Y. (2006).
  [Discrete Differential Forms for Computational Modeling](https://dl.acm.org/doi/pdf/10.1145/1185657.1185665)
- Blair Perot, J. & Zusi, C. (2014).
  [Differential forms for scientists and engineers ](https://www.sciencedirect.com/science/article/pii/S0021999113005354)
- Crane, K., de Goes, F., Desbrun, M. & Schröder, P. (2013).
  [Digital geometry processing with discrete exterior calculus ](https://dl.acm.org/doi/pdf/10.1145/2504435.2504442)

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise,
any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.

[repo]: https://github.com/m0lentum/dexterior
[pydec]: https://github.com/hirani/pydec/
