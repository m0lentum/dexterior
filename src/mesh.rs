//! The core discretization structure of DEC, the (simplicial) mesh.

/// Low-level mesh construction and corresponding tests.
mod mesh_construction;
/// re-export the testing meshes for use in other modules' tests
/// (pub because it's also used in examples at the moment.
/// this should be changed once we have better mesh generation tools)
#[doc(hidden)]
pub use mesh_construction::{tiny_mesh_2d, tiny_mesh_3d};

//

use fixedbitset as fb;
use nalgebra as na;
use nalgebra_sparse as nas;

use itertools::izip;
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
pub struct SimplexCollection<const DIM: usize> {
    /// points per simplex in the storage Vec
    simplex_size: usize,
    /// indices stored in a flat Vec to avoid generics for dimension
    indices: Vec<usize>,
    /// matrix where the rows correspond to DIM-simplices
    /// and the columns to (DIM-1) simplices.
    /// this matrix is the coboundary operator for (DIM-1) simplices,
    /// but it's stored with DIM-simplices
    /// because its rows can be efficiently used to navigate to boundary simplices.
    boundary_map: nas::CsrMatrix<Orientation>,
    /// transpose of the DIM+1-dimensional collection's `boundary_map`,
    /// stored separately for efficient access.
    /// the rows in this correspond to DIM-simplices again,
    /// and the columns to DIM+1-simplices.
    coboundary_map: nas::CsrMatrix<Orientation>,
    /// simplices on the boundary of the mesh.
    mesh_boundary: fb::FixedBitSet,
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
            boundary_map: nas::CsrMatrix::zeros(0, 0),
            coboundary_map: nas::CsrMatrix::zeros(0, 0),
            mesh_boundary: fb::FixedBitSet::default(),
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
            boundaries: self.boundary_map.row(idx),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundarySimplex {
    /// index of the simplex in the storage
    index: usize,
    /// orientation of the simplex relative to the simplex it bounds
    orientation: Orientation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orientation {
    #[default]
    Forward = 1,
    Backward = -1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubsetRef<'a, Dimension, Primality> {
    pub indices: &'a fb::FixedBitSet,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
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
    /// for each `DIM`-simplex in the mesh.
    ///
    /// See the [module-level docs][crate#operators] for usage information.
    pub fn new_zero_cochain<const DIM: usize, Primality>(&self) -> crate::Cochain<DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        let primal_dim = if Primality::IS_PRIMAL {
            DIM
        } else {
            MESH_DIM - DIM
        };
        crate::Cochain::zeros(self.simplex_count_dyn(primal_dim))
    }

    /// Construct an exterior derivative operator.
    ///
    /// See the [module-level docs][crate#operators] for usage information.
    pub fn d<const DIM: usize, Primality>(&self) -> crate::ExteriorDerivative<DIM, Primality>
    where
        na::Const<DIM>: na::DimNameAdd<na::U1>,
        na::Const<MESH_DIM>:
            na::DimNameSub<na::DimNameSum<na::Const<DIM>, na::U1>> + na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        let orientation_mat = if Primality::IS_PRIMAL {
            &self.simplices[DIM + 1].boundary_map
        } else {
            &self.simplices[MESH_DIM - DIM - 1].coboundary_map
        };
        // build the same matrix
        // but with Orientations converted to floats for easy multiplication.
        // technically we could reference `orientation_mat` directly in the operator,
        // but that would require introducing lifetimes to operators which gets annoying quickly
        let float_mat = nas::CsrMatrix::try_from_pattern_and_values(
            orientation_mat.pattern().clone(),
            orientation_mat
                .values()
                .iter()
                .map(|o| *o as isize as f64)
                .collect(),
        )
        .unwrap();
        crate::ExteriorDerivative::new(float_mat)
    }

    /// Construct a Hodge star operator.
    ///
    /// See the [module-level docs][crate#operators] for usage information.
    pub fn star<const DIM: usize, Primality>(&self) -> crate::HodgeStar<DIM, MESH_DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        let primal_dim = if Primality::IS_PRIMAL {
            DIM
        } else {
            MESH_DIM - DIM
        };
        let simplex_count = self.simplex_count_dyn(primal_dim);

        let compute_diag_val = if Primality::IS_PRIMAL {
            |(primal_vol, dual_vol): (&f64, &f64)| *primal_vol / *dual_vol
        } else {
            |(primal_vol, dual_vol): (&f64, &f64)| *dual_vol / *primal_vol
        };

        let diag = na::DVector::from_iterator(
            simplex_count,
            izip!(
                &self.simplices[primal_dim].volumes,
                &self.simplices[primal_dim].dual_volumes
            )
            .map(compute_diag_val),
        );

        crate::HodgeStar::new(diag)
    }

    pub fn boundary<const DIM: usize>(&self) -> SubsetRef<'_, na::Const<DIM>, Primal>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        SubsetRef {
            indices: &self.simplices[DIM].mesh_boundary,
            _marker: std::marker::PhantomData,
        }
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
/// so contents are hidden from docs.
pub trait MeshPrimality {
    /// Constant for runtime branching.
    ///
    /// There used to be some GAT magic and associated methods here,
    /// but branching on this is much easier to read
    /// and should optimize to roughly the same machine code.
    #[doc(hidden)]
    const IS_PRIMAL: bool;
    /// Maps Primal to Dual and Dual to Primal.
    #[doc(hidden)]
    type Opposite: MeshPrimality;
}

impl MeshPrimality for Primal {
    const IS_PRIMAL: bool = true;
    type Opposite = Dual;
}

impl MeshPrimality for Dual {
    const IS_PRIMAL: bool = false;
    type Opposite = Primal;
}

//
// iterators and views
//

// note: these are extremely unfinished.
// the eventual goal is to be able to access all the simplex information
// (circumcenter, volume etc.) as well as boundaries and coboundaries
// through these views, and ideally this should be available even through IterMut

pub struct SimplexView<'a> {
    indices: &'a [usize],
    boundaries: nas::csr::CsrRow<'a, Orientation>,
}

pub struct SimplexViewMut<'a> {
    indices: &'a mut [usize],
    boundaries: &'a mut [BoundarySimplex],
}

pub struct SimplexIter<'a, const MESH_DIM: usize> {
    simplices: &'a [SimplexCollection<MESH_DIM>],
    dim: usize,
    index: usize,
    len: usize,
}

impl<'a, const MESH_DIM: usize> Iterator for SimplexIter<'a, MESH_DIM> {
    type Item = SimplexView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        if self.index >= self.len {
            return None;
        }
        Some(self.simplices[self.dim].get(self.index))
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
