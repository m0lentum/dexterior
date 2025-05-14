//! Utilities for interpolating cochain values on a mesh.

use crate::{Cochain, Dual, DualCellView, MatrixOperator, Primal, SimplicialMesh};

use itertools::{izip, Itertools};
use nalgebra as na;
use nalgebra_sparse as nas;

/// Create an interpolation operator that maps a dual 0-cochain to a primal one.
///
/// NOTE: This is an experimental feature, currently only implemented for 2D meshes.
/// Generalizing to higher dimensions requires some care and has not been needed yet.
///
/// This is done using the generalized barycentric coordinates
/// of `Warren et al. (2007). Barycentric coordinates for convex sets.`
/// The method reproduces linear functions accurately
/// everywhere except on the mesh boundary,
/// where the dual cells corresponding to vertices are incomplete.
pub fn dual_to_primal(
    mesh: &SimplicialMesh<2>,
) -> MatrixOperator<Cochain<0, Dual>, Cochain<0, Primal>> {
    // reusable buffer to hold dual vertices and their weights for one dual cell
    let mut dual_vertices: Vec<(DualCellView<na::U0, 2>, f64)> = Vec::new();
    // matrix to hold finalized weight values
    let mut op_matrix = nas::CooMatrix::new(mesh.simplex_count::<0>(), mesh.simplex_count::<2>());

    for vert in mesh.simplices::<0>() {
        let vert_pos = vert.vertices().next().unwrap();
        // first we need to find each vertex of the corresponding dual cell
        for (_, edge) in vert.coboundary() {
            for (_, tri) in edge.coboundary() {
                if !dual_vertices.iter().any(|(c, _)| c == &tri.dual()) {
                    dual_vertices.push((tri.dual(), 0.));
                }
            }
        }

        // for each vertex of the dual cell, find the adjacent edge normals
        // (which are the tangent vectors of the corresponding triangle edges)
        // and use these to compute the weight of that vertex at the primal vertex
        let mut total_weight = 0.;
        for (dv, weight) in &mut dual_vertices {
            let dv_pos = dv.dual().circumcenter();

            let mut normals = [na::Vector2::<f64>::zeros(); 2];
            for (i, (_, edge)) in dv
                .dual()
                .boundary()
                .filter(|(_, e)| e.vertex_indices().contains(&vert.index()))
                .enumerate()
            {
                let mut edge_verts = edge.vertices();
                let edge_verts = [edge_verts.next().unwrap(), edge_verts.next().unwrap()];
                normals[i] = (edge_verts[1] - edge_verts[0]).normalize();
                // ensure the normal points outward from the dual cell
                if edge.vertex_indices().next().unwrap() != vert.index() {
                    normals[i] = -normals[i];
                }
            }

            // weight formula from the paper cited in the doc comment
            let num = f64::abs(normals[0].x * normals[1].y - normals[0].y * normals[1].x);
            let denom: f64 = normals.iter().map(|n| (dv_pos - vert_pos).dot(n)).product();
            *weight = num / denom;
            total_weight += *weight;
        }

        // normalize using the total weight
        for (dv, weight) in &mut dual_vertices {
            op_matrix.push(vert.index(), dv.index(), *weight / total_weight);
        }

        dual_vertices.clear();
    }

    let op_matrix = nas::CsrMatrix::from(&op_matrix);
    MatrixOperator::from(op_matrix)
}

/// Create an operator that maps a primal 0-cochain to a dual 0-cochain
/// using barycentric interpolation.
pub fn primal_to_dual<const DIM: usize>(
    mesh: &SimplicialMesh<DIM>,
) -> MatrixOperator<Cochain<0, Primal>, Cochain<0, Dual>>
where
    na::Const<DIM>: na::DimNameSub<na::Const<DIM>>,
    na::Const<DIM>: na::DimNameSub<na::U0>,
{
    let mut op_matrix = nas::CooMatrix::new(mesh.simplex_count::<DIM>(), mesh.simplex_count::<0>());

    // we need to collect vertices into a buffer to pass to `barycentric_coordinates`;
    // reuse one allocation for the whole thing
    let mut vertex_buf = Vec::with_capacity(DIM + 1);
    for dim_simp in mesh.simplices::<DIM>() {
        vertex_buf.extend(dim_simp.vertices());

        let bary_coords = barycentric_coords(&vertex_buf, dim_simp.circumcenter());
        for (vert_idx, bary) in izip!(dim_simp.vertex_indices(), bary_coords.iter()) {
            op_matrix.push(dim_simp.index(), vert_idx, *bary);
        }

        vertex_buf.clear();
    }

    MatrixOperator::from(nas::CsrMatrix::from(&op_matrix))
}

/// Compute barycentric coordinates for a point with respect to a DIM-simplex.
///
/// Returns a dynamically sized vector with dimension
/// corresponding to the number of points in the simplex.
/// A statically-sized vector is difficult to provide here because it requires adding 1 to DIM,
/// which makes the types very difficult (not entirely sure it's even possible).
/// This isn't called in a hot loop so it should be fine for perf.
///
/// Simplices with dimension less than DIM are not currently supported
/// because the possibility of the point not being coplanar with the simplex
/// adds some additional complexity that has not been needed so far.
fn barycentric_coords<const DIM: usize>(
    simplex: &[na::SVector<f64, DIM>],
    point: na::SVector<f64, DIM>,
) -> na::DVector<f64> {
    assert!(simplex.len() == DIM + 1, "Unsupported simplex dimension");

    let mut ret = na::DVector::zeros(simplex.len());

    // this could be a statically sized matrix
    // but that leads to some strange trait bound issues
    let mut num_mat = na::DMatrix::zeros(DIM, DIM);
    let mut denom_mat = num_mat.clone();
    for coord_i in 0..simplex.len() {
        for (col_i, simp_i) in (0..simplex.len()).filter(|i| *i != coord_i).enumerate() {
            num_mat.set_column(col_i, &(point - simplex[simp_i]));
            denom_mat.set_column(col_i, &(simplex[coord_i] - simplex[simp_i]));
        }
        let num = num_mat.determinant();
        let denom = denom_mat.determinant();
        // we can assume a non-degenerate simplex here (and thus denom != 0)
        // because this has been checked in the mesh construction process.
        // note also that denom is the same between each coordinate,
        // but we recompute it here to make its sign match the numerator's
        // and thus handle cases where the point is outside the simplex.
        // this isn't the most efficient but should be fine for our purposes
        ret[coord_i] = num / denom;
    }

    ret
}

//
// tests
//

#[cfg(test)]
mod tests {
    use itertools::izip;

    use super::*;
    use crate::{
        mesh::{tiny_mesh_2d, tiny_mesh_3d},
        Cochain, Dual,
    };

    #[test]
    fn interpolate_linear_precision() {
        let mesh = tiny_mesh_2d();

        let dual_primal = dual_to_primal(&mesh);
        let primal_dual = primal_to_dual(&mesh);
        // test the linear precision property of barycentric coordinates
        // by asserting that the linear coordinate functions are reproduced correctly
        // (only in the mesh interior for dual-to-primal.
        // TODO: can we guarantee anything about the truncated cells at the boundary?)
        let x_coords_primal: Cochain<0, Primal> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[0]));
        let y_coords_primal: Cochain<0, Primal> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[1]));
        let x_coords_dual: Cochain<0, Dual> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[0]));
        let y_coords_dual: Cochain<0, Dual> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[1]));

        let x_interp_dtp = &dual_primal * &x_coords_dual;
        let y_interp_dtp = &dual_primal * &y_coords_dual;
        let x_interp_ptd = &primal_dual * &x_coords_primal;
        let y_interp_ptd = &primal_dual * &y_coords_primal;

        // the mesh only has one interior vertex, located at the origin
        assert!(
            x_interp_dtp.values[3].abs() < 1e-10,
            "Actual x-coords: {:?}\nInterpolated: {:?}",
            x_coords_primal.values,
            x_interp_dtp.values,
        );
        assert!(
            y_interp_dtp.values[3].abs() < 1e-10,
            "Actual y-coords: {:?}\nInterpolated: {:?}",
            y_coords_primal.values,
            y_interp_dtp.values,
        );

        // same for primal-dual; this one also works near the boundary
        for (interp, actual) in izip!(&x_interp_ptd.values, &x_coords_dual.values) {
            assert!(
                (interp - actual).abs() < 1e-10,
                "Actual x-coords: {:?}\nInterpolated: {:?}",
                x_coords_dual.values,
                x_interp_ptd.values,
            );
        }
        for (interp, actual) in izip!(&y_interp_ptd.values, &y_coords_dual.values) {
            assert!(
                (interp - actual).abs() < 1e-10,
                "Actual y-coords: {:?}\nInterpolated: {:?}",
                y_coords_dual.values,
                y_interp_ptd.values,
            );
        }

        // test primal-to-dual also in 3d
        // (dual-to-primal currently isn't implemented)

        let mesh = tiny_mesh_3d();
        let primal_dual = primal_to_dual(&mesh);

        let x_coords_primal: Cochain<0, Primal> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[0]));
        let y_coords_primal: Cochain<0, Primal> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[1]));
        let x_coords_dual: Cochain<0, Dual> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[0]));
        let y_coords_dual: Cochain<0, Dual> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[1]));

        let x_interp_ptd = &primal_dual * &x_coords_primal;
        let y_interp_ptd = &primal_dual * &y_coords_primal;

        for (interp, actual) in izip!(&x_interp_ptd.values, &x_coords_dual.values) {
            assert!(
                (interp - actual).abs() < 1e-10,
                "Actual x-coords: {:?}\nInterpolated: {:?}",
                x_coords_dual.values,
                x_interp_ptd.values,
            );
        }
        for (interp, actual) in izip!(&y_interp_ptd.values, &y_coords_dual.values) {
            assert!(
                (interp - actual).abs() < 1e-10,
                "Actual y-coords: {:?}\nInterpolated: {:?}",
                y_coords_dual.values,
                y_interp_ptd.values,
            );
        }
    }

    #[test]
    fn barycentric_coords_are_correct() {
        let simplex_2d = [
            na::SVector::<f64, 2>::new(-1., 0.),
            na::SVector::<f64, 2>::new(1., 0.),
            na::SVector::<f64, 2>::new(0., 1.),
        ];
        // test a few different points
        // by expressing them in barycentric coordinates,
        // computing the corresponding point in global coordinates,
        // and computing the barycentric coordinates for it
        let barys_2d = [
            na::SVector::<f64, 3>::new(1., 0., 0.),
            na::SVector::<f64, 3>::new(0., 1., 0.),
            na::SVector::<f64, 3>::new(0., 0., 1.),
            na::SVector::<f64, 3>::new(1., 0., 1.),
            na::SVector::<f64, 3>::new(1., 1., 1.),
            na::SVector::<f64, 3>::new(0.5, 0.25, 0.75),
            // this should also work for points outside the simplex
            na::SVector::<f64, 3>::new(-1., 1., 1.),
            na::SVector::<f64, 3>::new(0., -1., 2.),
            na::SVector::<f64, 3>::new(5., 1., 0.),
        ]
        .map(|bary| bary / bary.iter().sum::<f64>());

        for bary in &barys_2d {
            let point: na::SVector<f64, 2> = izip!(bary.iter(), &simplex_2d)
                .map(|(weight, simp_point)| *weight * simp_point)
                .sum();
            let computed_bary = barycentric_coords::<2>(&simplex_2d, point);
            let computed_sum = computed_bary.iter().sum::<f64>();
            assert!(
                (computed_sum - 1.).abs() < 1e-10,
                "Sum was not 1: {computed_sum}\n(actual: {bary}, computed: {computed_bary})"
            );
            assert!(
                (&computed_bary - bary).magnitude() < 1e-10,
                "actual: {bary}, computed: {computed_bary}"
            );
        }

        let simplex_3d = [
            na::SVector::<f64, 3>::new(-1., 0., 0.),
            na::SVector::<f64, 3>::new(1., 0., 0.),
            na::SVector::<f64, 3>::new(0., 1., 0.),
            na::SVector::<f64, 3>::new(0., 0., 1.),
        ];
        let barys_3d = [
            na::SVector::<f64, 4>::new(1., 0., 0., 0.),
            na::SVector::<f64, 4>::new(0., 1., 0., 0.),
            na::SVector::<f64, 4>::new(0., 0., 1., 0.),
            na::SVector::<f64, 4>::new(1., 0., 1., 0.),
            na::SVector::<f64, 4>::new(1., 1., 1., 1.),
            na::SVector::<f64, 4>::new(0.5, 0.25, 0.75, 0.25),
            na::SVector::<f64, 4>::new(0.25, 0.3, 0.11, 0.9),
            na::SVector::<f64, 4>::new(-1., 1., 1., 1.),
            na::SVector::<f64, 4>::new(0., -1., 2., 0.),
        ]
        .map(|bary| bary / bary.iter().sum::<f64>());

        for bary in &barys_3d {
            let point: na::SVector<f64, 3> = izip!(bary.iter(), &simplex_3d)
                .map(|(weight, simp_point)| *weight * simp_point)
                .sum();
            let computed_bary = barycentric_coords::<3>(&simplex_3d, point);
            let computed_sum = computed_bary.iter().sum::<f64>();
            assert!(
                (computed_sum - 1.).abs() < 1e-10,
                "Sum was not 1: {computed_sum}\n(actual: {bary}, computed: {computed_bary})"
            );
            assert!(
                (&computed_bary - bary).magnitude() < 1e-10,
                "actual: {bary}, computed: {computed_bary}"
            );
        }

        let simplex_4d = [
            na::SVector::<f64, 4>::new(-1., 0., 0., 0.),
            na::SVector::<f64, 4>::new(1., 0., 0., 0.),
            na::SVector::<f64, 4>::new(0., 1., 0., 0.),
            na::SVector::<f64, 4>::new(0., 0., 1., 0.),
            na::SVector::<f64, 4>::new(0., 0., 0., 1.),
        ];
        let barys_4d = [
            na::SVector::<f64, 5>::new(1., 0., 0., 0., 0.),
            na::SVector::<f64, 5>::new(0., 1., 0., 0., 0.),
            na::SVector::<f64, 5>::new(0., 0., 1., 0., 0.),
            na::SVector::<f64, 5>::new(1., 0., 1., 0., 0.),
            na::SVector::<f64, 5>::new(1., 1., 1., 1., 1.),
            na::SVector::<f64, 5>::new(0.5, 0.25, 0.75, 0.25, 0.5),
            na::SVector::<f64, 5>::new(0.25, 0.3, 0.11, 0.9, 0.1),
            na::SVector::<f64, 5>::new(-1., 1., 1., 1., -1.),
            na::SVector::<f64, 5>::new(0., -1., 2., 0., 1.),
            na::SVector::<f64, 5>::new(5., -1., 1., 0.5, 0.),
        ]
        .map(|bary| bary / bary.iter().sum::<f64>());

        for bary in &barys_4d {
            let point: na::SVector<f64, 4> = izip!(bary.iter(), &simplex_4d)
                .map(|(weight, simp_point)| *weight * simp_point)
                .sum();
            let computed_bary = barycentric_coords::<4>(&simplex_4d, point);
            let computed_sum = computed_bary.iter().sum::<f64>();
            assert!(
                (computed_sum - 1.).abs() < 1e-10,
                "Sum was not 1: {computed_sum}\n(actual: {bary}, computed: {computed_bary})"
            );
            assert!(
                (&computed_bary - bary).magnitude() < 1e-10,
                "actual: {bary}, computed: {computed_bary}"
            );
        }
    }
}
