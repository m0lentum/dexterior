//! The core discretization structure of DEC, the (simplicial) mesh.

/// Low-level mesh construction and corresponding tests.
mod mesh_construction;
/// re-export the testing meshes for use in other modules' tests
/// (pub because it's also used in examples at the moment.
/// this should be changed once we have better mesh generation tools)
#[doc(hidden)]
pub use mesh_construction::{tiny_mesh_2d, tiny_mesh_3d};

//

use nalgebra as na;
use nalgebra_sparse as nas;

use std::rc::Rc;

/// A DEC mesh where the primal cells are all simplices
/// (points, line segments, triangles, tetrahedra etc).
#[derive(Clone, Debug)]
pub struct SimplicialMesh<const DIM: usize> {
    /// vertices stored in a Rc so that they can be accessed from multiple locations.
    /// mutation is (at least for now) not necessary.
    vertices: Rc<[na::SVector<f64, DIM>]>,
    /// storage for each dimension of simplex in the mesh
    /// (except 0, as those are just the vertices)
    simplices: Vec<SimplexCollection<DIM>>,
}

#[derive(Clone, Debug)]
struct SimplexCollection<const DIM: usize> {
    /// points per simplex in the storage Vec
    simplex_size: usize,
    /// indices stored in a flat Vec to avoid generics for dimension
    indices: Vec<usize>,
    /// boundary simplices on the next level down
    boundaries: Vec<BoundarySimplex>,
    /// circumcenters Rc'd so that 0-simplices
    /// can have the mesh vertices here without duplicating data
    circumcenters: Rc<[na::SVector<f64, DIM>]>,
    /// unsigned volumes of the primal simplices
    volumes: Vec<f64>,
    /// unsigned volumes of the corresponding dual simplices
    dual_volumes: Vec<f64>,
}

impl<const DIM: usize> Default for SimplexCollection<DIM> {
    fn default() -> Self {
        // initialize empty collections
        // which will be filled during mesh construction
        Self {
            simplex_size: 0,
            indices: Vec::new(),
            boundaries: Vec::new(),
            circumcenters: Rc::from([]),
            volumes: Vec::new(),
            dual_volumes: Vec::new(),
        }
    }
}

impl<const DIM: usize> SimplexCollection<DIM> {
    /// Get the number of simplices in the collection.
    #[inline]
    fn len(&self) -> usize {
        self.indices.len() / self.simplex_size
    }

    fn get(&self, idx: usize) -> SimplexView<'_> {
        let start_idx = idx * self.simplex_size;
        let idx_range = start_idx..start_idx + self.simplex_size;
        SimplexView {
            indices: &self.indices[idx_range.clone()],
            boundaries: if self.simplex_size == 1 {
                &[]
            } else {
                &self.boundaries[idx_range.clone()]
            },
        }
    }

    fn iter(&self) -> SimplexIter<'_> {
        SimplexIter {
            index_iter: self.indices.chunks_exact(self.simplex_size),
            // 0-simplices are a special case that does not have boundaries
            boundary_iter: if self.simplex_size == 1 {
                self.boundaries.chunks_exact(0)
            } else {
                self.boundaries.chunks_exact(self.simplex_size)
            },
        }
    }

    fn iter_mut(&mut self) -> SimplexIterMut<'_> {
        SimplexIterMut {
            index_iter: self.indices.chunks_exact_mut(self.simplex_size),
            boundary_iter: if self.simplex_size == 1 {
                self.boundaries.chunks_exact_mut(0)
            } else {
                self.boundaries.chunks_exact_mut(self.simplex_size)
            },
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

impl<const MESH_DIM: usize> SimplicialMesh<MESH_DIM> {
    /// Construct a mesh from raw vertices and indices.
    ///
    /// The indices are given as a flat array,
    /// where every `DIM + 1` indices correspond to one `DIM`-simplex.
    #[inline]
    pub fn new(vertices: Vec<na::SVector<f64, MESH_DIM>>, indices: Vec<usize>) -> Self {
        mesh_construction::build_mesh(vertices, indices)
    }

    /// Get the number of `DIM`-simplices in the mesh.
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
        self.simplices[dim].len()
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
        let simplices = &self.simplices[input_dim + 1];

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
