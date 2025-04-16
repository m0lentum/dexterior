//! Utilities for interpolating cochain values on a mesh.

use crate::{Cochain, ComposedOperator, Dual, DualCellView, Primal, SimplicialMesh};

use itertools::Itertools;
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
) -> ComposedOperator<Cochain<0, Dual>, Cochain<0, Primal>> {
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
    ComposedOperator::from(op_matrix)
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mesh::tiny_mesh_2d, Cochain, Dual};

    #[test]
    fn interpolate_2d() {
        let mesh = tiny_mesh_2d();

        let interp = dual_to_primal(&mesh);
        // test the linear precision property of barycentric coordinates
        // by asserting that the linear coordinate functions
        // are reproduced correctly in the mesh interior
        // (TODO: can we guarantee anything about the truncated cells at the boundary?)
        let x_coords: Cochain<0, Dual> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[0]));
        let y_coords: Cochain<0, Dual> =
            mesh.integrate_cochain(crate::quadrature::Pointwise(|p| p[1]));

        let x_interp = &interp * &x_coords;
        let y_interp = &interp * &y_coords;

        // the mesh only has one interior vertex, located at the origin
        assert!(
            x_interp.values[3].abs() < 1e-6,
            "Actual x-coords: {:?}\nInterpolated: {:?}",
            mesh.vertices().iter().map(|v| v.x).collect::<Vec<_>>(),
            x_interp.values,
        );
        assert!(
            y_interp.values[3].abs() < 1e-6,
            "Actual y-coords: {:?}\nInterpolated: {:?}",
            mesh.vertices().iter().map(|v| v.y).collect::<Vec<_>>(),
            y_interp.values,
        );
    }
}
