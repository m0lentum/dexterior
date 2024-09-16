//! Operations for integrating functions over simplices.
//!
//! These are used with
//! [`SimplicialMesh::integrate_cochain`][crate::SimplicialMesh::integrate_cochain].
//!
//! For 0-dimensional "integration" use [`Pointwise`],
//! which simply evaluates the given function at each point.
//!
//! For 1-dimensional line integrals we have Gauss-Legendre quadratures
//! of a few different orders ([`GaussLegendre3`], [`GaussLegendre6`], [`GaussLegendre9`]).
//! Each of these is exact for polynomials of order `2n-1` or less.
//! Pick one that is accurate enough for your use case.
//!
//! Higher-dimensional quadratures are currently not implemented,
//! but you can bring your own with [`Manual`].
//! However, this hasn't been tested and may not work correctly
//! for dimensions greater than 1.
//! See
//! [`SimplicialMesh::integrate_cochain`][crate::SimplicialMesh::integrate_cochain]
//! for details.
//!
//! # Examples
//!
//! ```
//! # use dexterior_core::{mesh::tiny_mesh_2d, quadrature::{Pointwise, GaussLegendre3}, Cochain, Primal};
//! # let mesh = tiny_mesh_2d();
//! let my_scalar_field = |pos: nalgebra::Vector2<f64>| {
//!     2. * pos.x.powi(5) + pos.y.powi(3)
//! };
//! let my_vector_field = |pos: nalgebra::Vector2<f64>| {
//!     nalgebra::Vector2::new(pos.x * pos.y, 2. * pos.x)
//! };
//! // create a 0-cochain by evaluating a scalar field at vertices
//! let c0: Cochain<0, Primal> = mesh.integrate_cochain(Pointwise(my_scalar_field));
//! // create a 1-cochain by integrating a scalar field over line segments
//! let c1_scalar: Cochain<1, Primal> = mesh.integrate_cochain(GaussLegendre3(
//!     |pos, _dir| my_scalar_field(pos)
//! ));
//! // create a 1-cochain by integrating a vector field over line segments
//! let c1_vector: Cochain<1, Primal> = mesh.integrate_cochain(GaussLegendre3(
//!     |pos, dir| my_vector_field(pos).dot(&dir)
//! ));
//! ```

use nalgebra as na;

/// An algorithm for computing the integral of a function over a cell.
pub trait Quadrature<const DIM: usize, const MESH_DIM: usize> {
    /// Compute the approximate integral given the vertices of a cell.
    fn compute(&self, vertices: &[na::SVector<f64, MESH_DIM>]) -> f64;
}

/// Use your own algorithm for integration.
///
/// The function given to this takes the vertices of a `DIM`-simplex as a parameter
/// and returns the integral over the simplex.
#[derive(Clone)]
pub struct Manual<const DIM: usize, const MESH_DIM: usize, IntgFn>(pub IntgFn)
where
    IntgFn: Fn(&[na::SVector<f64, MESH_DIM>]) -> f64;

impl<const DIM: usize, const MESH_DIM: usize, IntgFn> Quadrature<DIM, MESH_DIM>
    for Manual<DIM, MESH_DIM, IntgFn>
where
    IntgFn: Fn(&[na::SVector<f64, MESH_DIM>]) -> f64,
{
    fn compute(&self, vertices: &[na::SVector<f64, MESH_DIM>]) -> f64 {
        self.0(vertices)
    }
}

/// "Integrate" a 0-cochain by evaluating the function at each vertex.
///
/// See the [module-level docs][self] for examples.
#[derive(Clone)]
pub struct Pointwise<const MESH_DIM: usize, IntgFn>(pub IntgFn)
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>) -> f64;

impl<const MESH_DIM: usize, IntgFn> Quadrature<0, MESH_DIM> for Pointwise<MESH_DIM, IntgFn>
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>) -> f64,
{
    fn compute(&self, vertices: &[na::SVector<f64, MESH_DIM>]) -> f64 {
        assert!(vertices.len() == 1, "Mismatched quadrature dimension");
        self.0(vertices[0])
    }
}

// Gauss-Legendre quadratures for 1D simplices.
// It's probably enough for our purposes
// to just have a few hardcoded orders instead of a general solution
//
// source for the weights:
// https://pomax.github.io/bezierinfo/legendre-gauss.html

#[derive(Clone, Copy)]
struct GLPoint {
    weight: f64,
    abscissa: f64,
}

const WEIGHTS_GL_3: [GLPoint; 3] = [
    GLPoint {
        weight: 0.8888888888888888,
        abscissa: 0.0000000000000000,
    },
    GLPoint {
        weight: 0.5555555555555556,
        abscissa: -0.7745966692414834,
    },
    GLPoint {
        weight: 0.5555555555555556,
        abscissa: 0.7745966692414834,
    },
];

const WEIGHTS_GL_6: [GLPoint; 6] = [
    GLPoint {
        weight: 0.3607615730481386,
        abscissa: 0.6612093864662645,
    },
    GLPoint {
        weight: 0.3607615730481386,
        abscissa: -0.6612093864662645,
    },
    GLPoint {
        weight: 0.467913934572691,
        abscissa: -0.2386191860831969,
    },
    GLPoint {
        weight: 0.467913934572691,
        abscissa: 0.2386191860831969,
    },
    GLPoint {
        weight: 0.1713244923791704,
        abscissa: -0.932469514203152,
    },
    GLPoint {
        weight: 0.1713244923791704,
        abscissa: 0.932469514203152,
    },
];

const WEIGHTS_GL_9: [GLPoint; 9] = [
    GLPoint {
        weight: 0.3302393550012598,
        abscissa: 0.0000000000000000,
    },
    GLPoint {
        weight: 0.1806481606948574,
        abscissa: -0.8360311073266358,
    },
    GLPoint {
        weight: 0.1806481606948574,
        abscissa: 0.8360311073266358,
    },
    GLPoint {
        weight: 0.0812743883615744,
        abscissa: -0.9681602395076261,
    },
    GLPoint {
        weight: 0.0812743883615744,
        abscissa: 0.9681602395076261,
    },
    GLPoint {
        weight: 0.3123470770400029,
        abscissa: -0.3242534234038089,
    },
    GLPoint {
        weight: 0.3123470770400029,
        abscissa: 0.3242534234038089,
    },
    GLPoint {
        weight: 0.2606106964029354,
        abscissa: -0.6133714327005904,
    },
    GLPoint {
        weight: 0.2606106964029354,
        abscissa: 0.6133714327005904,
    },
];

fn gauss_legendre<const MESH_DIM: usize>(
    start: na::SVector<f64, MESH_DIM>,
    end: na::SVector<f64, MESH_DIM>,
    weights: &[GLPoint],
    f: impl Fn(na::SVector<f64, MESH_DIM>, na::Unit<na::SVector<f64, MESH_DIM>>) -> f64,
) -> f64 {
    let center = (start + end) / 2.;
    let half_segment = (end - start) / 2.;
    let half_length = half_segment.magnitude();
    let dir = na::Unit::new_unchecked(half_segment / half_length);
    weights
        .iter()
        .map(|p| p.weight * f(center + p.abscissa * half_segment, dir))
        .sum::<f64>()
        * half_length
}

/// Third-order Gauss-Legendre quadrature for integrating a function over a line segment.
///
/// This is exact for polynomials up to degree 5.
/// The parameters for the function
/// are a position in space and the direction vector of the line segment being integrated over.
/// See the [module-level docs][self] for examples.
#[derive(Clone)]
pub struct GaussLegendre3<const MESH_DIM: usize, IntgFn>(pub IntgFn)
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>, na::Unit<na::SVector<f64, MESH_DIM>>) -> f64;

impl<const MESH_DIM: usize, IntgFn> Quadrature<1, MESH_DIM> for GaussLegendre3<MESH_DIM, IntgFn>
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>, na::Unit<na::SVector<f64, MESH_DIM>>) -> f64,
{
    fn compute(&self, vertices: &[na::SVector<f64, MESH_DIM>]) -> f64 {
        assert!(vertices.len() == 2, "Mismatched quadrature dimension");
        gauss_legendre(vertices[0], vertices[1], &WEIGHTS_GL_3, &self.0)
    }
}

/// Sixth-order Gauss-Legendre quadrature for integrating a function over a line segment.
///
/// This is exact for polynomials up to degree 11.
/// The parameters for the function
/// are a position in space and the direction vector of the line segment being integrated over.
/// See the [module-level docs][self] for examples.
#[derive(Clone)]
pub struct GaussLegendre6<const MESH_DIM: usize, IntgFn>(pub IntgFn)
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>, na::Unit<na::SVector<f64, MESH_DIM>>) -> f64;

impl<const MESH_DIM: usize, IntgFn> Quadrature<1, MESH_DIM> for GaussLegendre6<MESH_DIM, IntgFn>
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>, na::Unit<na::SVector<f64, MESH_DIM>>) -> f64,
{
    fn compute(&self, vertices: &[na::SVector<f64, MESH_DIM>]) -> f64 {
        assert!(vertices.len() == 2, "Mismatched quadrature dimension");
        gauss_legendre(vertices[0], vertices[1], &WEIGHTS_GL_6, &self.0)
    }
}

/// Ninth-order Gauss-Legendre quadrature for integrating a function over a line segment.
///
/// This is exact for polynomials up to degree 17.
/// The parameters for the function
/// are a position in space and the direction vector of the line segment being integrated over.
/// See the [module-level docs][self] for examples.
#[derive(Clone)]
pub struct GaussLegendre9<const MESH_DIM: usize, IntgFn>(pub IntgFn)
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>, na::Unit<na::SVector<f64, MESH_DIM>>) -> f64;

impl<const MESH_DIM: usize, IntgFn> Quadrature<1, MESH_DIM> for GaussLegendre9<MESH_DIM, IntgFn>
where
    IntgFn: Fn(na::SVector<f64, MESH_DIM>, na::Unit<na::SVector<f64, MESH_DIM>>) -> f64,
{
    fn compute(&self, vertices: &[na::SVector<f64, MESH_DIM>]) -> f64 {
        assert!(vertices.len() == 2, "Mismatched quadrature dimension");
        gauss_legendre(vertices[0], vertices[1], &WEIGHTS_GL_9, &self.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::{mesh::tiny_mesh_2d, mesh::tiny_mesh_3d, Cochain, Dual, Primal, SimplicialMesh};

    use super::*;

    /// Check that the Gauss-Legendre quadratures
    /// are actually accurate for the degrees of polynomial they're supposed to be.
    #[test]
    fn gauss_legendre() {
        let mesh_2d = tiny_mesh_2d();
        let mesh_3d = tiny_mesh_3d();

        // coefficients for an arbitrary degree 5 polynomial in the x coordinate,
        // which should be able to be integrated exactly with GL3.
        // (these could be fuzzed for more rigorous testing,
        // but I'm pretty confident the quadratures are correct
        // even with just one test case getting exact results)
        let deg_5_coefs = [6., 2.3, -3., 1., -1., 0.2];
        // same for degrees 11 and 17 for GL6 and 9
        let deg_11_coefs = [1.5, -8., 2.1, 6.6, -2., 0., 1.1, -3.85, -2.5, 0.2, -0.2, 1.];
        let deg_17_coefs = [
            -2., 1.1, 1.3, 0., -0.5, 6.8, -10., 2.25, -0.05, 1.44, 4.2, -6.7, 0., 1.15, -3.2, 5.55,
            1., -2.5,
        ];

        fn test_polynomial<const MESH_DIM: usize>(mesh: &SimplicialMesh<MESH_DIM>, coefs: &[f64])
        where
            na::Const<MESH_DIM>: na::DimNameSub<na::Const<1>>,
        {
            let polynomial = |p: na::SVector<f64, MESH_DIM>| -> f64 {
                coefs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| p[0].powi(i as i32) * c)
                    .sum()
            };
            // compare against the analytically computed integral
            let antiderivative = |p: na::SVector<f64, MESH_DIM>| -> f64 {
                coefs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| p[0].powi(i as i32 + 1) * c / (i + 1) as f64)
                    .sum()
            };

            let quad_analytic = Manual(|p| {
                let dir = na::Unit::new_normalize(p[1] - p[0]);
                if dir[0] == 0. {
                    // for edges perfectly aligned with the y axis this becomes a constant function
                    polynomial(p[0]) * (p[1] - p[0]).magnitude()
                } else {
                    (antiderivative(p[1]) - antiderivative(p[0])) / dir[0]
                }
            });

            // which version of GL we use here turns out
            // to be a really annoying thing to abstract out,
            // so we've got a bit of repetition here
            let c_num: Cochain<1, Primal> = if coefs.len() <= 6 {
                mesh.integrate_cochain(GaussLegendre3(|p, _| polynomial(p)))
            } else if coefs.len() <= 12 {
                mesh.integrate_cochain(GaussLegendre6(|p, _| polynomial(p)))
            } else {
                mesh.integrate_cochain(GaussLegendre9(|p, _| polynomial(p)))
            };
            let c_an: Cochain<1, Primal> = mesh.integrate_cochain(quad_analytic.clone());
            assert!(
                (&c_an - &c_num).values.magnitude_squared() < f64::EPSILON,
                "Analytic and numeric results disagree (primal)\nAnalytic: {c_an:?}\nNumeric: {c_num:?}",
            );

            // same test on the dual edges
            let c_num: Cochain<1, Dual> = if coefs.len() <= 5 {
                mesh.integrate_cochain(GaussLegendre3(|p, _| polynomial(p)))
            } else if coefs.len() <= 12 {
                mesh.integrate_cochain(GaussLegendre6(|p, _| polynomial(p)))
            } else {
                mesh.integrate_cochain(GaussLegendre9(|p, _| polynomial(p)))
            };
            let c_an: Cochain<1, Dual> = mesh.integrate_cochain(quad_analytic);
            assert!(
                (&c_an - &c_num).values.magnitude_squared() < f64::EPSILON,
                "Analytic and numeric results disagree (dual)\nAnalytic: {c_an:?}\nNumeric: {c_num:?}",
            );
        }

        println!("2d mesh, GL3");
        test_polynomial(&mesh_2d, &deg_5_coefs);
        println!("3d mesh, GL3");
        test_polynomial(&mesh_3d, &deg_5_coefs);
        println!("2d mesh, GL6");
        test_polynomial(&mesh_2d, &deg_11_coefs);
        println!("3d mesh, GL6");
        test_polynomial(&mesh_3d, &deg_11_coefs);
        println!("2d mesh, GL9");
        test_polynomial(&mesh_2d, &deg_17_coefs);
        println!("3d mesh, GL9");
        test_polynomial(&mesh_3d, &deg_17_coefs);
    }
}
