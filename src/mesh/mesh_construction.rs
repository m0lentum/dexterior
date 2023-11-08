use nalgebra as na;

use itertools::{iproduct, izip};
use std::rc::Rc;

use super::{BoundarySimplex, Orientation, SimplexCollection, SimplicialMesh};

/// Construct a mesh from raw vertices and indices.
///
/// The indices are given as a flat array,
/// where every `DIM + 1` indices correspond to one `DIM`-simplex.
pub fn build_mesh<const MESH_DIM: usize>(
    vertices: Vec<na::SVector<f64, MESH_DIM>>,
    indices: Vec<usize>,
) -> SimplicialMesh<MESH_DIM> {
    let vertices: Rc<[na::SVector<f64, MESH_DIM>]> = Rc::from(vertices);
    // collection for every dimension of simplex, including 0
    // (even though those are just the vertices),
    // for unified storage and iteration
    let mut simplices: Vec<SimplexCollection<MESH_DIM>> = (0..=MESH_DIM)
        .map(|i| SimplexCollection {
            simplex_size: i + 1,
            ..Default::default()
        })
        .collect();
    // circumcenters will go inside the simplex collections,
    // but they need to be constructed as Vecs first and transferred into Rcs at the end
    let mut circumcenters: [Vec<na::SVector<f64, MESH_DIM>>; MESH_DIM] =
        std::array::from_fn(|_| Vec::new());

    // the collection of 0-simplices is just the vertices in order
    // with volume 1
    simplices[0].indices = (0..vertices.len()).collect();
    simplices[0].circumcenters = vertices.clone();
    simplices[0].volumes.resize(vertices.len(), 1.0);

    //
    // compute sub-simplices
    //

    // highest dimension simplices have the indices given as parameter
    simplices[MESH_DIM].indices = indices;
    // by convention, sort simplices to have their indices in ascending order.
    // this simplifies things by enabling a consistent way
    // to identify a simplex with its vertices
    for simplex in simplices[MESH_DIM].indices.chunks_exact_mut(MESH_DIM + 1) {
        simplex.sort_unstable();
    }

    // rest of the levels (excluding 0-simplices, hence skip(1))
    // are inferred from boundaries of the top-level simplices
    let mut level_iter = simplices.iter_mut().skip(1).rev().peekable();
    while let Some(upper_simplices) = level_iter.next() {
        if upper_simplices.simplex_size == 1 {
            break;
        }
        // preallocate space for boundary elements
        upper_simplices
            .boundaries
            .resize(upper_simplices.indices.len(), BoundarySimplex::default());

        let Some(lower_simplices) = level_iter.peek_mut() else {
            // we're at the 1-simplex level, where boundary simplices are vertices.
            // set boundary indices and break
            for simplex in upper_simplices.iter_mut() {
                simplex.boundaries[0] = BoundarySimplex {
                    index: simplex.indices[0],
                    orientation: Orientation::Backward,
                };
                simplex.boundaries[1] = BoundarySimplex {
                    index: simplex.indices[1],
                    orientation: Orientation::Forward,
                };
            }

            break;
        };

        // buffer to hold the simplex currently being processed.
        // we don't push directly into lower_simplices.indices
        // in order to check if a simplex already exists
        let mut curr_simplex: Vec<usize> = Vec::with_capacity(lower_simplices.simplex_size);

        for upper_simplex in upper_simplices.iter_mut() {
            // every unique combination of vertices in the upper simplex
            // is a simplex on its boundary
            for exclude_idx in 0..upper_simplex.indices.len() {
                curr_simplex.clear();
                for (i, vert_id) in upper_simplex.indices.iter().enumerate() {
                    if i != exclude_idx {
                        curr_simplex.push(*vert_id);
                    }
                }

                // boundary orientations alternate between forward and backward
                // when defined in this order.
                // see Discrete Differential Forms for Computational Modeling by Desbrun et al. (2006)
                // https://dl.acm.org/doi/pdf/10.1145/1185657.1185665
                let orientation = if exclude_idx % 2 == 0 {
                    Orientation::Forward
                } else {
                    Orientation::Backward
                };

                // linear search through already added simplices to deduplicate.
                // this isn't the most efficient for large meshes,
                // but we'll optimize later if it becomes a problem
                let already_existing_lower = lower_simplices
                    // lower_simplices.iter() doesn't work yet
                    // because the boundary elements aren't allocated until next loop;
                    // manually construct iterator over indices instead
                    .indices
                    .chunks_exact(lower_simplices.simplex_size)
                    .enumerate()
                    .find(|(_, s)| *s == curr_simplex);

                if let Some((found_idx, _)) = already_existing_lower {
                    upper_simplex.boundaries[exclude_idx] = BoundarySimplex {
                        index: found_idx,
                        orientation,
                    };
                } else {
                    let new_idx = lower_simplices.len();
                    lower_simplices.indices.extend_from_slice(&curr_simplex);
                    upper_simplex.boundaries[exclude_idx] = BoundarySimplex {
                        index: new_idx,
                        orientation,
                    };
                }
            }
        }
    }

    //
    // compute circumcenters
    //

    // temporarily store barycentric coordinates for the circumcenters
    // for use in dual volume computation (signs of elementary dual volumes).
    // similar to indices, these are stored as flat Vecs of floats
    // because each has a different dimension.
    // the first dimension that has these stored is 2
    // because for 1-simplices it's always (0.5, 0.5).
    let mut circumcenter_bary_coords: Vec<Vec<f64>> = vec![Vec::new(); MESH_DIM - 1];

    // for 1-simplices (line segments) the circumcenter is simply the midpoint,
    // compute those as a special case for efficiency

    let simplices_1 = &mut simplices[1];
    let indices_1 = &simplices_1.indices;
    for indices in indices_1.chunks_exact(2) {
        let verts = [vertices[indices[0]], vertices[indices[1]]];
        // note that circumcenters don't exist for 0-simplices,
        // so addressing here is off by one
        circumcenters[0].push(0.5 * (verts[0] + verts[1]));
    }

    // for the rest, solve for the circumcenter in barycentric coordinates
    // using the linear system from the PyDEC paper
    // (https://dl.acm.org/doi/pdf/10.1145/2382585.2382588, section 10.1)
    for (simplices, circumcenters, bary_coords) in izip!(
        &mut simplices[2..],
        &mut circumcenters[1..],
        &mut circumcenter_bary_coords
    ) {
        let indices = &simplices.indices;
        // dimension is simplex_size + 1 because there's an extra row
        // for normalizing the barycentric coordinates
        let system_dim = simplices.simplex_size + 1;
        let mut coef_mat = na::DMatrix::zeros(system_dim, system_dim);
        let mut rhs = na::DVector::zeros(system_dim);
        // fill in the constant last row and column first
        for i in 0..system_dim - 1 {
            coef_mat[(i, system_dim - 1)] = 1.0;
            coef_mat[(system_dim - 1, i)] = 1.0;
        }
        rhs[system_dim - 1] = 1.0;

        // solve for each simplex, reusing the allocated matrices
        for indices in indices.chunks_exact(simplices.simplex_size) {
            for row in 0..simplices.simplex_size {
                let row_vert = vertices[indices[row]];
                rhs[row] = row_vert.dot(&row_vert);
                for col in 0..simplices.simplex_size {
                    let col_vert = vertices[indices[col]];
                    coef_mat[(row, col)] = 2.0 * row_vert.dot(&col_vert);
                }
            }
            // a decomposition is needed to solve the linear system
            // and it seems nalgebra's APIs don't allow reusing allocated memory for it.
            // shouldn't be a big deal to reallocate this for every simplex,
            // we'll optimize later if it becomes an issue

            // TODO: return an error instead of panicking if the system is singular
            let bary = coef_mat
                .clone()
                .lu()
                .solve(&rhs)
                .expect("Degenerate simplex");
            // compute circumcenter in cartesian coordinates
            // from the barycentric coordinates obtained from the linear system
            let circumcenter = izip!(bary.iter(), indices)
                .map(|(&bary_weight, &idx)| bary_weight * vertices[idx])
                .sum();
            circumcenters.push(circumcenter);

            for bc in bary.iter().take(bary.len() - 1) {
                bary_coords.push(*bc);
            }
        }
    }

    // move the circumcenters into the Rcs in SimplexCollections

    for (simplices, circumcenters) in izip!(&mut simplices[1..], circumcenters) {
        simplices.circumcenters = Rc::from(circumcenters);
    }

    //
    // compute primal volumes
    //

    // again, simplified special case for line segments
    let simplices_1 = &mut simplices[1];
    let indices_1 = &simplices_1.indices;
    for indices in indices_1.chunks_exact(2) {
        let verts = [vertices[indices[0]], vertices[indices[1]]];
        simplices_1.volumes.push((verts[1] - verts[0]).magnitude());
    }

    // for the rest, compute the volume as the determinant of a matrix
    // vol = sqrt(det(V^T V)) / p!
    // (see the PyDEC paper section 10.1)

    // this could be done more efficiently
    // by writing the determinant formula by hand and exploiting symmetry,
    // but I'll do that after I get it working if I can be bothered to
    for simplices in &mut simplices[2..] {
        let indices = &simplices.indices;

        let edge_count = simplices.simplex_size - 1;
        // the term `p!` in the volume formula
        let edge_count_factorial: usize = (1..=edge_count).product();
        // again, reusing a matrix allocation.
        // this is the square matrix V^T V which we compute manually
        let mut det_mat = na::DMatrix::zeros(edge_count, edge_count);
        // also reusing space for the edge vectors which are the columns of V
        let mut edges: Vec<na::SVector<f64, MESH_DIM>> = vec![na::SVector::zeros(); edge_count];

        for indices in indices.chunks_exact(simplices.simplex_size) {
            // edges are vectors from the first vertex in the simplex to the other ones
            for edge_idx in 0..edge_count {
                edges[edge_idx] = vertices[indices[edge_idx + 1]] - vertices[indices[0]];
            }
            // fill in the square matrix
            for (row, col) in iproduct!(0..edge_count, 0..edge_count) {
                det_mat[(row, col)] = edges[row].dot(&edges[col]);
            }
            // compute volume from the square matrix
            let vol = f64::sqrt(det_mat.determinant()) / edge_count_factorial as f64;
            simplices.volumes.push(vol);
        }
    }

    //
    // compute dual volumes
    //

    // initialize volumes to zero first,
    // we'll accumulate volumes from multiple "elementary duals" into each
    for simplices in &mut simplices {
        simplices.dual_volumes.resize(simplices.volumes.len(), 0.0);
    }
    // ..except the dual vertices which have a volume of 1
    for dv in &mut simplices[MESH_DIM].dual_volumes {
        *dv = 1.0;
    }

    // we'll be operating on simplices of all dimensions here
    // so we have to use indices for borrow checker reasons
    for top_simplex_idx in 0..simplices[MESH_DIM].len() {
        // for each top-dimensional simplex,
        // generate the "first circumcentric subdivision"
        // (see Desbrun et al. (2005). Discrete Exterior Calculus, chapter 3)
        // and add each of its simplices to the corresponding dual volumes

        let top_center = simplices[MESH_DIM].circumcenters[top_simplex_idx];
        let start_idx = top_simplex_idx * simplices[MESH_DIM].simplex_size;
        for boundary_idx in 0..simplices[MESH_DIM].simplex_size {
            // 1-dimensional elementary duals as a non-recursive special case
            // because their volumes are easy to compute
            let boundary_simplex_idx =
                simplices[MESH_DIM].boundaries[start_idx + boundary_idx].index;
            let bound_center = simplices[MESH_DIM - 1].circumcenters[boundary_simplex_idx];
            let edge = bound_center - top_center;

            // sign of the elementary dual volume is determined by
            // whether the previous circumcenter is in the same halfspace
            // relative to the boundary simplex as the opposite vertex.
            // this can be checked by looking at the sign
            // of the barycentric coordinate of the opposite vertex.
            // see Hirani et al. (2012). Delaunay Hodge Star
            // https://www.sciencedirect.com/science/article/pii/S0010448512002436
            let sign = if MESH_DIM <= 1 {
                1.0
            } else {
                // the indexing here works because each boundary simplex is constructed by omitting
                // the `n`th vertex of the upper level simplex.
                // barycentric coordinates for 0- and 1-simplices are constant
                // and thus omitted from the cache, hence `MESH_DIM - 2`.
                let opposite_bary =
                    circumcenter_bary_coords[MESH_DIM - 2][start_idx + boundary_idx];
                opposite_bary.signum()
            };

            simplices[MESH_DIM - 1].dual_volumes[boundary_simplex_idx] +=
                edge.magnitude().copysign(sign);

            // recursively traverse the rest of the simplex dimensions,
            // using the determinant formula that was also used for primal volumes
            // (see the primal volume section for explanatory comments)
            // to compute volumes of the elementary duals

            // reusing allocations again.
            // the top left part of this matrix stays constant,
            // take advantage of that by only setting it once
            let mut vol_mat = na::DMatrix::zeros(MESH_DIM, MESH_DIM);
            vol_mat[(0, 0)] = edge.dot(&edge);
            let mut edges: Vec<na::SVector<f64, MESH_DIM>> = Vec::new();
            edges.push(edge);

            // named parameters for the `traverse` function to keep them readable.
            // lots of stuff here because we need to explicitly pass
            // variables from the surrounding function
            // (can't use a closure because closures can't be recursive)
            struct TraversalState<'a, const MESH_DIM: usize> {
                // dimension of the primal simplex whose boundaries we'll look at next
                curr_dim: usize,
                // index of the aforementioned simplex
                curr_simplex_idx: usize,
                // sign of the current elementary dual's volume
                curr_sign: f64,
                // denominator in the volume formula
                edge_count_factorial: usize,
                // simplex collections of the mesh
                simplices: &'a mut [SimplexCollection<MESH_DIM>],
                // edges of the elementary dual simplex being processed
                edges: &'a mut Vec<na::SVector<f64, MESH_DIM>>,
                // reusable matrix allocation for volume computations
                vol_mat: &'a mut na::DMatrix<f64>,
                // circumcenter of the top-level simplex we started from
                root_vertex: &'a na::SVector<f64, MESH_DIM>,
                // barycentric coordinates of circumcenters
                // of 2- and higher-dimensional simplices
                circumcenter_bary_coords: &'a [Vec<f64>],
            }

            fn traverse<const MESH_DIM: usize>(s: TraversalState<'_, MESH_DIM>) {
                if s.curr_dim == 0 {
                    return;
                }

                // using the number of edges in the array
                // to keep track of the dimension of the dual simplex we're working on
                let dual_dim = s.edges.len();
                // add the point count of the current simplex dimension to the denominator
                let next_factorial = s.edge_count_factorial * (dual_dim + 1);

                let start_idx = s.curr_simplex_idx * s.simplices[s.curr_dim].simplex_size;
                for boundary_idx in 0..s.simplices[s.curr_dim].simplex_size {
                    let boundary_simplex_idx =
                        s.simplices[s.curr_dim].boundaries[start_idx + boundary_idx].index;
                    let bound_center =
                        s.simplices[s.curr_dim - 1].circumcenters[boundary_simplex_idx];
                    let new_edge = bound_center - s.root_vertex;
                    s.edges.push(new_edge);
                    // update the bottom right section of the volume matrix
                    s.vol_mat[(dual_dim, dual_dim)] = new_edge.dot(&new_edge);
                    for edge_idx in 0..dual_dim {
                        let dot_val = s.edges[edge_idx].dot(&new_edge);
                        s.vol_mat[(edge_idx, dual_dim)] = dot_val;
                        s.vol_mat[(dual_dim, edge_idx)] = dot_val;
                    }

                    let vol = f64::sqrt(
                        s.vol_mat
                            .view_range(0..=dual_dim, 0..=dual_dim)
                            .determinant(),
                    ) / next_factorial as f64;

                    // see earlier comment about sign
                    let next_sign = if s.curr_dim <= 1 {
                        s.curr_sign
                    } else {
                        let opposite_bary =
                            s.circumcenter_bary_coords[s.curr_dim - 2][start_idx + boundary_idx];
                        s.curr_sign * opposite_bary.signum()
                    };

                    s.simplices[s.curr_dim - 1].dual_volumes[boundary_simplex_idx] +=
                        vol.copysign(next_sign);

                    traverse(TraversalState {
                        curr_dim: s.curr_dim - 1,
                        curr_simplex_idx: boundary_simplex_idx,
                        curr_sign: next_sign,
                        edge_count_factorial: next_factorial,
                        simplices: s.simplices,
                        edges: s.edges,
                        vol_mat: s.vol_mat,
                        root_vertex: s.root_vertex,
                        circumcenter_bary_coords: s.circumcenter_bary_coords,
                    });

                    s.edges.pop();
                }
            }

            traverse(TraversalState {
                curr_dim: MESH_DIM - 1,
                curr_simplex_idx: boundary_simplex_idx,
                curr_sign: sign,
                edge_count_factorial: 1,
                simplices: &mut simplices,
                edges: &mut edges,
                vol_mat: &mut vol_mat,
                root_vertex: &top_center,
                circumcenter_bary_coords: &circumcenter_bary_coords,
            });
        }
    }

    // sort simplex boundaries by index.
    // this lets us build exterior derivative matrices more efficiently,
    // as the structure directly corresponds to a CSR matrix.
    // NOTE: this needs to be done after dual volume computation
    // because dual volumes make use of the original ordering.
    for simplices in &mut simplices {
        for boundaries in simplices
            .boundaries
            .chunks_exact_mut(simplices.simplex_size)
        {
            boundaries.sort_unstable_by_key(|b| b.index);
        }
    }

    SimplicialMesh {
        vertices,
        simplices,
    }
}

//
// tests
//

type Vec2 = na::SVector<f64, 2>;
type Vec3 = na::SVector<f64, 3>;

/// A small hexagon-shaped 2D mesh for testing basic functionality.
/// Shaped somewhat like this:
///    ____
///   /\  /\
///  /__\/__\
///  \  /\  /
///   \/__\/
///
/// with vertices and triangles ordered left to right, top to bottom.
///
/// This is public for visibility in doctests, which frequently need an instance of a mesh.
/// It is not meant to be used by users and thus hidden from docs.
/// Eventually there should be a mesh generator API that can replace this.
#[doc(hidden)]
pub fn tiny_mesh_2d() -> SimplicialMesh<2> {
    let vertices = vec![
        Vec2::new(-0.5, 1.0),
        Vec2::new(0.5, 1.0),
        Vec2::new(-1.0, 0.0),
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(-0.5, -1.0),
        Vec2::new(0.5, -1.0),
    ];
    #[rustfmt::skip]
        let indices = vec![
            0, 2, 3,
            0, 1, 3,
            1, 3, 4,
            2, 3, 5,
            3, 5, 6,
            3, 4, 6,
        ];
    SimplicialMesh::new(vertices, indices)
}

/// A small 3D mesh for testing basic functionality.
/// Four tetrahedra arranged into a diamond shape,
/// split like this down the x,y plane:
///
///    /\
///   /__\
///   \  /
///    \/
///
/// and with a single point both up and down the z-axis.
///
/// This is public for visibility in doctests, which frequently need an instance of a mesh.
/// It is not meant to be used by users and thus hidden from docs.
/// Eventually there should be a mesh generator API that can replace this.
#[doc(hidden)]
pub fn tiny_mesh_3d() -> SimplicialMesh<3> {
    let vertices = vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    #[rustfmt::skip]
        let indices = vec![
            0, 1, 2, 4,
            0, 1, 2, 5,
            1, 2, 3, 4,
            1, 2, 3, 5,
        ];

    SimplicialMesh::new(vertices, indices)
}

// Tests here are concerned with the mesh structure being constructed correctly.
// For tests on exterior derivative and Hodge star, see `operator.rs`
#[cfg(test)]
mod tests {
    use super::*;
    use approx::{abs_diff_eq, relative_eq};

    /// Lower-dimensional simplices, circumcenters, volumes etc.
    /// are generated correctly for a simple 2d mesh.
    ///
    /// Note: this may break spuriously if the order simplices are generated in is changed.
    /// If this happens, instead of manually fixing the order tested here,
    /// it may be better to modify this test to check the presence of simplices
    /// (and absence of unintended ones) independently of order.
    /// Boundaries become tricky if you do that,
    /// but should be doable by e.g. testing against vertex positions instead of indices.
    #[test]
    fn tiny_2d_mesh_is_correct() {
        let mesh = tiny_mesh_2d();

        // sub-simplices

        #[rustfmt::skip]
        let expected_1_simplices = vec![
            2,3, 0,3, 0,2,
            1,3, 0,1,
            3,4, 1,4,
            3,5, 2,5,
            5,6, 3,6,
            4,6,
        ];
        assert_eq!(
            expected_1_simplices, mesh.simplices[1].indices,
            "incorrect 1-simplices"
        );

        // boundaries

        // orientations as integers for brevity
        #[rustfmt::skip]
        let expected_2_boundaries = vec![
            (0, 1), (1, -1), (2, 1),
            (1, -1), (3, 1), (4, 1),
            (3, 1), (5, 1), (6, -1),
            (0, 1), (7, 1), (8, -1),
            (7, 1), (9, 1), (10, -1),
            (5, 1), (10, -1), (11, 1),
        ];
        let actual_2_boundaries: Vec<(usize, isize)> = mesh.simplices[2]
            .boundaries
            .iter()
            .map(|b| (b.index, b.orientation as isize))
            .collect();

        assert_eq!(
            expected_2_boundaries, actual_2_boundaries,
            "incorrect 2-simplex boundaries"
        );

        // primal volumes

        // all diagonal edges are the same length, as are all the horizontals
        let diag = f64::sqrt(5.0) / 2.0;
        let horiz = 1.0;
        #[rustfmt::skip]
        let expected_1_volumes = vec![
            horiz, diag, diag,
            diag, horiz,
            horiz, diag,
            diag, diag,
            horiz, diag,
            diag,
        ];
        let actual_1_volumes = &mesh.simplices[1].volumes;
        let all_approx_eq =
            izip!(&expected_1_volumes, actual_1_volumes).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected 1-volumes {expected_1_volumes:?}, got {actual_1_volumes:?}"
        );

        // all triangles are the same size (base 1, height 1)
        // this would be a more interesting test if they had different volumes,
        // but that would make other things harder
        let tri_vol = 0.5;
        let actual_2_volumes = &mesh.simplices[2].volumes;
        let all_correct_vol = actual_2_volumes.iter().all(|&v| v == tri_vol);
        assert!(
            all_correct_vol,
            "expected all 2-volumes {tri_vol}, got {actual_2_volumes:?}"
        );

        // circumcenters

        #[rustfmt::skip]
        let expected_1_centers: Vec<Vec2> = [
            (0.0, 1.0),
            (-0.75, 0.5), (-0.25, 0.5), (0.25, 0.5), (0.75, 0.5),
            (-0.5, 0.0), (0.5, 0.0),
            (-0.75, -0.5), (-0.25, -0.5), (0.25, -0.5), (0.75, -0.5),
            (0.0, -1.0),
        ]
        .into_iter()
        .map(|(x, y)| Vec2::new(x, y))
        .collect();

        let centers = &mesh.simplices[1].circumcenters;
        assert_eq!(expected_1_centers.len(), centers.len());

        for expected in expected_1_centers {
            let found = centers
                .iter()
                .any(|actual| (expected - actual).magnitude_squared() <= f64::EPSILON);
            assert!(
                found,
                "Expected 1-circumcenter {expected} not found in set {centers:?}"
            )
        }

        #[rustfmt::skip]
        let expected_2_centers: Vec<Vec2>  = [
            (-0.5, 0.375), (0.0, 0.625), (0.5, 0.375),
            (-0.5, -0.375), (0.0, -0.625), (0.5, -0.375),
        ]
        .into_iter()
        .map(|(x, y)| Vec2::new(x, y))
        .collect();

        let centers = &mesh.simplices[2].circumcenters;
        assert_eq!(expected_2_centers.len(), centers.len());

        for expected in expected_2_centers {
            let found = centers
                .iter()
                .any(|actual| (expected - actual).magnitude_squared() <= f64::EPSILON);
            assert!(
                found,
                "Expected 2-circumcenter {expected} not found in set {centers:?}"
            )
        }

        // dual volumes

        // dual diagonals also have the same lengths
        let dual_diag = f64::sqrt(5.0) / 4.0;
        #[rustfmt::skip]
        let expected_1_dual_vols = vec![
            0.75, dual_diag, 0.5 * dual_diag,
            dual_diag, 0.375,
            0.75, 0.5 * dual_diag,
            dual_diag, 0.5 * dual_diag,
            0.375, dual_diag,
            0.5 * dual_diag,
        ];
        let actual_1_dual_vols = &mesh.simplices[1].dual_volumes;
        let all_approx_eq =
            izip!(&expected_1_dual_vols, actual_1_dual_vols).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected dual 1-volumes {expected_1_dual_vols:?}, got {actual_1_dual_vols:?}"
        );

        // sizes of the elementary dual triangles
        let side_el = 5.0 / 64.0;
        let base_el = 3.0 / 32.0;
        // dual cells touching the top and bottom boundaries are the same shape
        let bound_vert = 3.0 * side_el + base_el;
        let bound_horiz = 2.0 * side_el + 2.0 * base_el;
        let center = 8.0 * side_el + 4.0 * base_el;
        #[rustfmt::skip]
        let expected_0_dual_vols = vec![
            bound_vert, bound_vert,
            bound_horiz, center, bound_horiz,
            bound_vert, bound_vert,
        ];

        let actual_0_dual_vols = &mesh.simplices[0].dual_volumes;
        let all_approx_eq =
            izip!(&expected_0_dual_vols, actual_0_dual_vols).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected dual 0-volumes {expected_0_dual_vols:?}, got {actual_0_dual_vols:?}"
        );
    }

    /// Lower-dimensional simplices, circumcenters, volumes etc.
    /// are generated correctly for a simple 3d mesh.
    ///
    /// Same note applies as the above test.
    #[test]
    fn tiny_3d_mesh_is_correct() {
        let mesh = tiny_mesh_3d();

        // sub-simplices

        #[rustfmt::skip]
        let expected_2_simplices = vec![
            1,2,4, 0,2,4, 0,1,4, 0,1,2,
            1,2,5, 0,2,5, 0,1,5,
            2,3,4, 1,3,4, 1,2,3,
            2,3,5, 1,3,5,
        ];
        assert_eq!(
            expected_2_simplices, mesh.simplices[2].indices,
            "incorrect 1-simplices"
        );

        #[rustfmt::skip]
        let expected_1_simplices = vec![
            2,4, 1,4, 1,2, 0,4, 0,2, 0,1,
            2,5, 1,5, 0,5,
            3,4, 2,3, 1,3,
            3,5,
        ];
        assert_eq!(
            expected_1_simplices, mesh.simplices[1].indices,
            "incorrect 2-simplices"
        );

        // boundaries

        #[rustfmt::skip]
        let expected_3_boundaries = vec![
            (0, 1), (1, -1), (2, 1), (3, -1),
            (3, -1), (4, 1), (5, -1), (6, 1),
            (0, 1), (7, 1), (8, -1), (9, -1),
            (4, 1), (9, -1), (10, 1), (11, -1),
        ];
        let actual_3_boundaries: Vec<(usize, isize)> = mesh.simplices[3]
            .boundaries
            .iter()
            .map(|b| (b.index, b.orientation as isize))
            .collect();

        assert_eq!(
            expected_3_boundaries, actual_3_boundaries,
            "incorrect 3-simplex boundaries"
        );

        // there are so many 2-simplex boundaries on this one
        // I can't be bothered to write them all out,
        // I'll trust the 2D test that these are correct for now.
        // should probably try to architect this test in a way
        // that doesn't explicitly list all indices

        // primal volumes

        // most diagonals are this length
        let diag = f64::sqrt(5.0) / 2.0;
        use std::f64::consts::SQRT_2;
        #[rustfmt::skip]
        let expected_1_volumes = vec![
            diag, diag, 1.0, SQRT_2, diag, diag,
            diag, diag, SQRT_2,
            SQRT_2, diag, diag,
            SQRT_2,
        ];
        let actual_1_volumes = &mesh.simplices[1].volumes;
        let all_approx_eq =
            izip!(&expected_1_volumes, actual_1_volumes).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected 1-volumes {expected_1_volumes:?}, got {actual_1_volumes:?}"
        );

        // triangles on the outer boundary all have the same area,
        // as do triangles inside of the mesh
        let inner = 0.5;
        // this one computed by hand on paper, just trust me bro
        let outer = diag * (f64::sqrt(30.0) / 5.0) / 2.0;
        let expected_2_volumes = vec![
            inner, outer, outer, inner, inner, outer, outer, outer, outer, inner, outer, outer,
        ];
        let actual_2_volumes = &mesh.simplices[2].volumes;
        let all_approx_eq =
            izip!(&expected_2_volumes, actual_2_volumes,).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected 2-volumes {expected_2_volumes:?}, got {actual_2_volumes:?}"
        );

        // all tetrahedra in this one have the same volume
        let tet_vol = 1.0 / 6.0;
        let actual_3_volumes = &mesh.simplices[3].volumes;
        let all_correct_vol = actual_3_volumes.iter().all(|&v| v == tet_vol);
        assert!(
            all_correct_vol,
            "expected all 3-volumes {tet_vol}, got {actual_3_volumes:?}"
        );

        // circumcenters

        #[rustfmt::skip]
        let expected_1_centers: Vec<Vec3> = [
            (-0.25, 0.5, 0.0), (0.25, 0.5, 0.0),
            (0.0, 0.0, 0.0),
            (-0.25, -0.5, 0.0), (0.25, -0.5, 0.0),
            (0.0, 0.5, 0.5), (0.0, -0.5, 0.5),
            (0.25, 0.0, 0.5), (-0.25, 0.0, 0.5),
            (0.0, 0.5, -0.5), (0.0, -0.5, -0.5),
            (0.25, 0.0, -0.5), (-0.25, 0.0, -0.5),
        ]
        .into_iter()
        .map(|(x, y, z)| Vec3::new(x, y, z))
        .collect();

        let centers = &mesh.simplices[1].circumcenters;
        assert_eq!(expected_1_centers.len(), centers.len());

        for expected in expected_1_centers {
            let found = centers
                .iter()
                .any(|actual| (expected - actual).magnitude_squared() <= f64::EPSILON);
            assert!(
                found,
                "Expected 1-circumcenter {expected} not found in set {centers:?}"
            )
        }

        #[rustfmt::skip]
        let expected_2_centers: Vec<Vec3> = [
            (0.0, 0.375, 0.0), (0.0, -0.375, 0.0),
            (0.0, 0.0, 0.375), (0.0, 0.0, -0.375),
            (0.08333, 0.41667, 0.41667), (0.08333, 0.41667, -0.41667),
            (0.08333, -0.41667, 0.41667), (0.08333, -0.41667, -0.41667),
            (-0.08333, -0.41667, 0.41667), (-0.08333, -0.41667, -0.41667),
            (-0.08333, 0.41667, 0.41667), (-0.08333, 0.41667, -0.41667),
        ]
        .into_iter()
        .map(|(x, y, z)| Vec3::new(x, y, z))
        .collect();

        let actual_2_centers = &mesh.simplices[2].circumcenters;
        assert_eq!(expected_2_centers.len(), actual_2_centers.len());

        for expected in expected_2_centers {
            let found = actual_2_centers
                .iter()
                .any(|actual| (expected - actual).magnitude_squared() <= 0.0001);
            assert!(
                found,
                "Expected 2-circumcenter {expected} not found in set {centers:?}"
            )
        }

        #[rustfmt::skip]
        let expected_3_centers: Vec<Vec3> = [
            (0.0, 0.375, 0.375), (0.0, 0.375, -0.375),
            (0.0, -0.375, 0.375), (0.0, -0.375, -0.375),
        ]
        .into_iter()
        .map(|(x, y, z)| Vec3::new(x, y, z))
        .collect();

        let actual_3_centers = &mesh.simplices[3].circumcenters;
        assert_eq!(expected_3_centers.len(), actual_3_centers.len());

        for expected in expected_3_centers {
            let found = actual_3_centers
                .iter()
                .any(|actual| (expected - actual).magnitude_squared() <= 0.0001);
            assert!(
                found,
                "Expected 3-circumcenter {expected} not found in set {centers:?}"
            )
        }

        // dual volumes

        // all triangles on the boundary have the same length dual edge,
        // same goes for all triangles inside the mesh
        let boundary = 0.10206;
        let inside = 0.75;
        #[rustfmt::skip]
        let expected_2_dual_vols = vec![
            inside, boundary, boundary, inside,
            inside, boundary, boundary,
            boundary, boundary, inside,
            boundary, boundary,
        ];
        let actual_2_dual_vols = &mesh.simplices[2].dual_volumes;
        let all_approx_eq = izip!(&expected_2_dual_vols, actual_2_dual_vols)
            .all(|(l, r)| abs_diff_eq!(l, r, epsilon = 0.00001));
        assert!(
            all_approx_eq,
            "expected dual 2-volumes {expected_2_dual_vols:?}, got {actual_2_dual_vols:?}"
        );

        #[rustfmt::skip]
        let expected_1_dual_vols = vec![
            0.1514008, 0.1514008, 0.75 * 0.75, 0.0147308, 0.1514008, 0.1514008,
            0.1514008, 0.1514008, 0.0147308,
            0.0147308, 0.1514008, 0.1514008,
            0.0147308,
        ];
        let actual_1_dual_vols = &mesh.simplices[1].dual_volumes;
        let all_approx_eq = izip!(&expected_1_dual_vols, actual_1_dual_vols)
            .all(|(l, r)| abs_diff_eq!(l, r, epsilon = 0.000001));
        assert!(
            all_approx_eq,
            "expected dual 1-volumes {expected_1_dual_vols:?}, got {actual_1_dual_vols:?}"
        );

        #[rustfmt::skip]
        let expected_0_dual_vols = vec![
            0.06336792, 0.20659792, 0.20659792, 0.06336792,
            0.06336792, 0.06336792,
        ];

        let actual_0_dual_vols = &mesh.simplices[0].dual_volumes;
        let all_approx_eq = izip!(&expected_0_dual_vols, actual_0_dual_vols)
            .all(|(l, r)| abs_diff_eq!(l, r, epsilon = 0.000001));
        assert!(
            all_approx_eq,
            "expected dual 0-volumes {expected_0_dual_vols:?}, got {actual_0_dual_vols:?}"
        );
    }

    /// Dual volumes are computed correctly for meshes
    /// with circumcenters outside of their simplices.
    #[test]
    fn non_well_centered_dual_volumes() {
        // diamond-shaped mesh with two triangles,
        // one of which is very shallow
        // with circumcenter inside the other one.
        let mesh_2d = SimplicialMesh::new(
            vec![
                Vec2::new(0.0, 0.5),
                Vec2::new(-1.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, -2.0),
            ],
            vec![0, 1, 2, 1, 2, 3],
        );

        let expected_1_dual_vols = [
            // this one should have one negative and one positive elementary dual
            0.0,
            f64::sqrt(5.0) / 2.0,
            f64::sqrt(5.0) / 2.0,
            f64::sqrt(5.0) / 4.0,
            f64::sqrt(5.0) / 4.0,
        ];
        let actual_1_dual_vols = &mesh_2d.simplices[1].dual_volumes;
        let all_approx_eq =
            izip!(&expected_1_dual_vols, actual_1_dual_vols).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected dual 1-volumes {expected_1_dual_vols:?}, got {actual_1_dual_vols:?}"
        );

        // the middle two should have one negative elementary dual
        // that cancels out another one (unsigned sum for them would be 1.375)
        let expected_0_dual_vols = [0.625, 0.625, 0.625, 0.625];
        let actual_0_dual_vols = &mesh_2d.simplices[0].dual_volumes;
        let all_approx_eq =
            izip!(&expected_0_dual_vols, actual_0_dual_vols).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected dual 0-volumes {expected_0_dual_vols:?}, got {actual_0_dual_vols:?}"
        );

        //
        // 3d
        //

        println!("3d");
        // two tetrahedra, both sharing a non-well-centered triangle
        // and one of them low enough to have a second negative sign in the volumes
        let mesh_3d = SimplicialMesh::new(
            vec![
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.5),
                Vec3::new(0.0, 0.0, -2.0),
            ],
            vec![0, 1, 2, 3, 0, 1, 2, 4],
        );

        // this one has circumcenters outside the entire mesh,
        // which generates some negative dual volumes
        let expected_2_dual_vols = [
            -0.75,
            4.0 / 3.0,
            4.0 / 3.0,
            // this is the triangle in the middle,
            // which has two identical elementary duals with opposite signs
            0.0,
            -0.75,
            0.9274260335029676,
            0.9274260335029676,
        ];
        let actual_2_dual_vols = &mesh_3d.simplices[2].dual_volumes;
        let all_approx_eq =
            izip!(&expected_2_dual_vols, actual_2_dual_vols).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected dual 2-volumes {expected_2_dual_vols:?}, got {actual_2_dual_vols:?}"
        );

        let expected_1_dual_vols = [
            0.124226 - 0.419263,
            0.124226 - 0.419263,
            0.0,
            0.628539,
            0.576763,
            0.576763,
            0.056568 - 0.209631,
            0.056568 - 0.209631,
            0.417219,
        ];
        let actual_1_dual_vols = &mesh_3d.simplices[1].dual_volumes;
        let all_approx_eq = izip!(&expected_1_dual_vols, actual_1_dual_vols)
            .all(|(l, r)| abs_diff_eq!(l, r, epsilon = 0.00001));
        assert!(
            all_approx_eq,
            "expected dual 1-volumes {expected_1_dual_vols:?}, got {actual_1_dual_vols:?}"
        );

        let expected_0_dual_vols = [0.432374, -0.004547, -0.004547, -0.035879, 0.029265];
        let actual_0_dual_vols = &mesh_3d.simplices[0].dual_volumes;
        let all_approx_eq = izip!(&expected_0_dual_vols, actual_0_dual_vols)
            .all(|(l, r)| abs_diff_eq!(l, r, epsilon = 0.00001));
        assert!(
            all_approx_eq,
            "expected dual 0-volumes {expected_0_dual_vols:?}, got {actual_0_dual_vols:?}"
        );
    }
}
