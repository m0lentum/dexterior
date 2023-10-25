//! The core data structure of DEC, the simplicial complex.

use nalgebra as na;
use nalgebra_sparse as nas;

use itertools::{iproduct, izip};

/// A DEC complex where the primal cells are all simplices
/// (points, line segments, triangles, tetrahedra etc).
#[derive(Clone, Debug)]
pub struct SimplicialComplex<const DIM: usize> {
    vertices: Vec<na::SVector<f64, DIM>>,
    /// storage for each dimension of simplex in the mesh
    /// (except 0, as those are just the vertices)
    simplices: Vec<SimplexCollection<DIM>>,
}

#[derive(Clone, Debug, Default)]
struct SimplexCollection<const DIM: usize> {
    /// points per simplex in the storage Vec
    simplex_size: usize,
    /// indices stored in a flat Vec to avoid generics for dimension
    indices: Vec<usize>,
    /// boundary simplices on the next level down
    boundaries: Vec<BoundarySimplex>,
    circumcenters: Vec<na::SVector<f64, DIM>>,
    volumes: Vec<f64>,
}

impl<const DIM: usize> SimplexCollection<DIM> {
    /// Get the number of simplices in the collection.
    #[inline]
    fn len(&self) -> usize {
        self.indices.len() / self.simplex_size
    }

    fn iter(&self) -> SimplexIter<'_> {
        SimplexIter {
            index_iter: self.indices.chunks_exact(self.simplex_size),
            boundary_iter: self.boundaries.chunks_exact(self.simplex_size),
        }
    }

    fn iter_mut(&mut self) -> SimplexIterMut<'_> {
        SimplexIterMut {
            index_iter: self.indices.chunks_exact_mut(self.simplex_size),
            boundary_iter: self.boundaries.chunks_exact_mut(self.simplex_size),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct BoundarySimplex {
    /// index of the simplex in the storage
    index: usize,
    /// orientation of the simplex relative to the simplex it bounds
    orientation: Orientation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Orientation {
    #[default]
    Forward = 1,
    Backward = -1,
}

impl<const MESH_DIM: usize> SimplicialComplex<MESH_DIM> {
    /// Construct a SimplicialComplex from raw vertices and indices.
    ///
    /// The indices are given as a flat array,
    /// where every `DIM + 1` indices correspond to one `DIM`-simplex.
    pub fn new(vertices: Vec<na::SVector<f64, MESH_DIM>>, indices: Vec<usize>) -> Self {
        let mut simplices: Vec<SimplexCollection<MESH_DIM>> = (0..MESH_DIM)
            .map(|i| SimplexCollection {
                // 0-simplices are omitted from these collections,
                // so the one at index 0 is the collection of 1-simplices,
                // which have 2 points each
                simplex_size: i + 2,
                ..Default::default()
            })
            .collect();

        //
        // compute sub-simplices
        //

        // highest dimension simplices have the indices given as parameter
        simplices[MESH_DIM - 1].indices = indices;
        // by convention, sort simplices to have their indices in ascending order.
        // this simplifies things by enabling a consistent way
        // to identify a simplex with its vertices
        for simplex in simplices[MESH_DIM - 1]
            .indices
            .chunks_exact_mut(MESH_DIM + 1)
        {
            simplex.sort_unstable();
        }

        // rest of the levels are inferred from boundaries of the top-level simplices
        let mut level_iter = simplices.iter_mut().rev().peekable();
        while let Some(upper_simplices) = level_iter.next() {
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

                // sort the simplex boundaries by index.
                // this lets us build exterior derivative matrices more efficiently,
                // as the structure directly corresponds to a CSR matrix
                // (TODO: should we already build the CSR matrix here
                // and not have this current structure at all?)
                upper_simplex.boundaries.sort_unstable_by_key(|b| b.index);
            }
        }

        //
        // compute circumcenters
        //

        // for 1-simplices (line segments) the circumcenter is simply the midpoint,
        // compute those as a special case for efficiency

        let simplices_1 = &mut simplices[0];
        let indices_1 = &simplices_1.indices;
        for indices in indices_1.chunks_exact(2) {
            let verts = [vertices[indices[0]], vertices[indices[1]]];
            simplices_1.circumcenters.push(0.5 * (verts[0] + verts[1]));
        }

        // for the rest, solve for the circumcenter in barycentric coordinates
        // using the linear system from the PyDEC paper
        // (https://dl.acm.org/doi/pdf/10.1145/2382585.2382588, section 10.1)
        for simplices in &mut simplices[1..] {
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
                simplices.circumcenters.push(circumcenter);
            }
        }

        //
        // compute primal volumes
        //

        // again, simplified special case for line segments
        let simplices_1 = &mut simplices[0];
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
        for simplices in &mut simplices {
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

        Self {
            vertices,
            simplices,
        }
    }

    /// Get the number of `DIM`-simplices in the complex.
    #[inline]
    pub fn simplex_count<const DIM: usize>(&self) -> usize
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        self.simplex_count_dyn(DIM)
    }

    /// Simplex count taking the dimension as a runtime parameter
    /// to allow usage in dynamic contexts (internal APIs)
    #[inline]
    fn simplex_count_dyn(&self, dim: usize) -> usize {
        // TODO: consider a more unified structure
        // where vertices also have a SimplexCollection
        // to simplify addressing and remove a likely source of off-by-one errors
        if dim == 0 {
            self.vertices.len()
        } else {
            self.simplices[dim - 1].len()
        }
    }

    /// Create a new cochain with a value of zero
    /// for each `Dimension`-simplex in the mesh.
    ///
    /// See the [module-level docs][crate#operators] for usage information.
    pub fn new_zero_cochain<const DIM: usize, Primality>(&self) -> crate::Cochain<DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        // GAT magic to compute the corresponding primal dimension at compile time
        let primal_dim =
            <Primality::PrimalDim<na::Const<DIM>, na::Const<MESH_DIM>> as na::DimName>::USIZE;
        crate::Cochain::zeros(self.simplex_count_dyn(primal_dim))
    }

    /// Construct an exterior derivative operator.
    ///
    /// See the [module-level docs][crate#operators] for usage information.
    pub fn d<const DIM: usize, Primality>(&self) -> crate::ExteriorDerivative<DIM, Primality>
    where
        na::Const<DIM>: na::DimNameAdd<na::U1>,
        na::Const<MESH_DIM>: na::DimNameSub<na::DimNameSum<na::Const<DIM>, na::U1>>,
        Primality: MeshPrimality,
    {
        let in_dim =
            <Primality::DInputPrimalDim<na::Const<DIM>, na::Const<MESH_DIM>> as na::DimName>::USIZE;
        let mat = Primality::convert_d_from_primal(self.build_coboundary_matrix(in_dim));
        crate::ExteriorDerivative::new(mat)
    }

    /// Construct a Hodge star operator.
    /// Not implemented correctly yet, currently just here to test APIs and operator composition.
    ///
    /// See the [module-level docs][crate#operators] for usage information.
    pub fn star<const DIM: usize, Primality>(&self) -> crate::HodgeStar<DIM, MESH_DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        let primal_dim =
            <Primality::PrimalDim<na::Const<DIM>, na::Const<MESH_DIM>> as na::DimName>::USIZE;
        let simplex_count = self.simplex_count_dyn(primal_dim);

        // TODO: replace this with the construction of an actual Hodge star

        let diag = na::DVector::repeat(simplex_count, 2.0);

        crate::HodgeStar::new(Primality::convert_star_from_primal(diag))
    }

    /// Constructs a coboundary matrix taking primal `dim`-cochains
    /// to primal `dim+1`-cochains.
    /// Internal API used by `Self::d`.
    fn build_coboundary_matrix(&self, input_dim: usize) -> nas::CsrMatrix<f64> {
        // no dimension check here, that is done by the generics in `d`

        // simplices of the output dimension
        let simplices = &self.simplices[input_dim];

        let row_count = simplices.len();
        let col_count = self.simplex_count_dyn(input_dim);

        // each row has the same number of nonzero elements,
        // so it's simple to construct the sparsity pattern of the matrix
        let row_offsets: Vec<usize> = (0..=row_count)
            .map(|i| i * simplices.simplex_size)
            .collect();
        let (col_indices, values): (Vec<usize>, Vec<f64>) = simplices
            .boundaries
            .iter()
            .map(|b| (b.index, b.orientation as isize as f64))
            .unzip();

        nas::CsrMatrix::try_from_csr_data(row_count, col_count, row_offsets, col_indices, values)
            .expect("Error in matrix construction. This is a bug in dexterior")
    }
}

//
// mesh primality generics
//

/// Marker type indicating a [`Cochain`][crate::Cochain]
/// or [`operator`][crate::operator] corresponds to a primal mesh.
#[derive(Clone, Copy, Debug)]
pub struct Primal;

/// Marker type indicating a [`Cochain`][crate::Cochain]
/// or [`operator`][crate::operator] corresponds to a dual mesh.
#[derive(Clone, Copy, Debug)]
pub struct Dual;

/// Trait allowing types and mesh methods to be generic
/// on whether they operate on the primal ([`Primal`][self::Primal])
/// or dual ([`Dual`][self::Dual]) mesh.
///
/// Not intended to be implemented by users,
/// so methods are hidden from docs.
pub trait MeshPrimality {
    /// Maps Primal to Dual and Dual to Primal.
    #[doc(hidden)]
    type Opposite: MeshPrimality;
    /// GAT that allows computing the corresponding primal dimension
    /// of a dual cochain at compile time.
    #[doc(hidden)]
    type PrimalDim<Dim: na::DimName, MeshDim: na::DimName + na::DimNameSub<Dim>>: na::DimName;
    /// GAT that allows computing the dimension of the primal exterior derivative
    /// that gets transposed into a dual one.
    #[doc(hidden)]
    type DInputPrimalDim<
        Dim: na::DimNameAdd<na::U1>,
        MeshDim: na::DimName + na::DimNameSub<na::DimNameSum<Dim, na::U1>>,
    >: na::DimName;

    /// Conversion procedure for exterior derivative constructed for the primal mesh.
    /// The exterior derivative on the dual mesh is the transpose
    /// of the one on the primal mesh.
    #[doc(hidden)]
    fn convert_d_from_primal(primal_d: nas::CsrMatrix<f64>) -> nas::CsrMatrix<f64>;

    /// Conversion procedure for Hodge star constructed for the primal mesh.
    /// The star on the dual mesh is the inverse of the one on the primal mesh.
    #[doc(hidden)]
    fn convert_star_from_primal(primal_diag: na::DVector<f64>) -> na::DVector<f64>;
}

impl MeshPrimality for Primal {
    type Opposite = Dual;
    type PrimalDim<Dim: na::DimName, MeshDim: na::DimName + na::DimNameSub<Dim>> = Dim;
    type DInputPrimalDim<
        Dim: na::DimNameAdd<na::U1>,
        MeshDim: na::DimName + na::DimNameSub<na::DimNameSum<Dim, na::U1>>,
    > = Dim;

    fn convert_d_from_primal(primal_d: nas::CsrMatrix<f64>) -> nas::CsrMatrix<f64> {
        primal_d
    }
    fn convert_star_from_primal(primal_diag: na::DVector<f64>) -> na::DVector<f64> {
        primal_diag
    }
}

impl MeshPrimality for Dual {
    type Opposite = Primal;
    type PrimalDim<Dim: na::DimName, MeshDim: na::DimName + na::DimNameSub<Dim>> =
        na::DimNameDiff<MeshDim, Dim>;
    type DInputPrimalDim<
        Dim: na::DimNameAdd<na::U1>,
        MeshDim: na::DimName + na::DimNameSub<na::DimNameSum<Dim, na::U1>>,
    > = na::DimNameDiff<MeshDim, na::DimNameSum<Dim, na::U1>>;

    fn convert_d_from_primal(primal_d: nas::CsrMatrix<f64>) -> nas::CsrMatrix<f64> {
        primal_d.transpose()
    }
    fn convert_star_from_primal(mut primal_diag: na::DVector<f64>) -> na::DVector<f64> {
        for elem in primal_diag.iter_mut() {
            *elem = 1.0 / *elem;
        }
        primal_diag
    }
}

//
// iterators and views
//

pub struct SimplexView<'a> {
    indices: &'a [usize],
    boundaries: &'a [BoundarySimplex],
}

pub struct SimplexViewMut<'a> {
    indices: &'a mut [usize],
    boundaries: &'a mut [BoundarySimplex],
}

pub struct SimplexIter<'a> {
    index_iter: std::slice::ChunksExact<'a, usize>,
    boundary_iter: std::slice::ChunksExact<'a, BoundarySimplex>,
}

impl<'a> Iterator for SimplexIter<'a> {
    type Item = SimplexView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let indices = self.index_iter.next()?;
        let boundaries = self.boundary_iter.next()?;
        Some(SimplexView {
            indices,
            boundaries,
        })
    }
}

struct SimplexIterMut<'a> {
    index_iter: std::slice::ChunksExactMut<'a, usize>,
    boundary_iter: std::slice::ChunksExactMut<'a, BoundarySimplex>,
}

impl<'a> Iterator for SimplexIterMut<'a> {
    type Item = SimplexViewMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let indices = self.index_iter.next()?;
        let boundaries = self.boundary_iter.next()?;
        Some(SimplexViewMut {
            indices,
            boundaries,
        })
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
pub fn tiny_mesh_2d() -> SimplicialComplex<2> {
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
    SimplicialComplex::new(vertices, indices)
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
pub fn tiny_mesh_3d() -> SimplicialComplex<3> {
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

    SimplicialComplex::new(vertices, indices)
}

// Tests here are concerned with the mesh structure being constructed correctly.
// For tests on exterior derivative and Hodge star, see `operator.rs`
#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;

    /// Lower-dimensional simplices and boundaries
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
            expected_1_simplices, mesh.simplices[0].indices,
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
        let actual_2_boundaries: Vec<(usize, isize)> = mesh.simplices[1]
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
        let actual_1_volumes = &mesh.simplices[0].volumes;
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
        let actual_2_volumes = &mesh.simplices[1].volumes;
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

        let centers = &mesh.simplices[0].circumcenters;
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

        let centers = &mesh.simplices[1].circumcenters;
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
    }

    /// Lower-dimensional simplices are generated correctly
    /// for a simple 3d mesh.
    ///
    /// Same note applies as the above test.
    #[test]
    fn simple_3d_simplices_are_correct() {
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
            expected_2_simplices, mesh.simplices[1].indices,
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
            expected_1_simplices, mesh.simplices[0].indices,
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
        let actual_3_boundaries: Vec<(usize, isize)> = mesh.simplices[2]
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
        let actual_1_volumes = &mesh.simplices[0].volumes;
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
        let actual_2_volumes = &mesh.simplices[1].volumes;
        let all_approx_eq =
            izip!(&expected_2_volumes, actual_2_volumes,).all(|(l, r)| relative_eq!(l, r));
        assert!(
            all_approx_eq,
            "expected 2-volumes {expected_2_volumes:?}, got {actual_2_volumes:?}"
        );

        // all tetrahedra in this one have the same volume
        let tet_vol = 1.0 / 6.0;
        let actual_3_volumes = &mesh.simplices[2].volumes;
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

        let centers = &mesh.simplices[0].circumcenters;
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

        let centers = &mesh.simplices[1].circumcenters;
        assert_eq!(expected_2_centers.len(), centers.len());

        for expected in expected_2_centers {
            let found = centers
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

        let centers = &mesh.simplices[2].circumcenters;
        assert_eq!(expected_3_centers.len(), centers.len());

        for expected in expected_3_centers {
            let found = centers
                .iter()
                .any(|actual| (expected - actual).magnitude_squared() <= 0.0001);
            assert!(
                found,
                "Expected 3-circumcenter {expected} not found in set {centers:?}"
            )
        }
    }
}
