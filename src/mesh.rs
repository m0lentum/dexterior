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
use std::{cell::OnceCell, rc::Rc};

/// A DEC mesh where the primal cells are all simplices
/// (points, line segments, triangles, tetrahedra etc).
#[derive(Clone, Debug)]
pub struct SimplicialMesh<const DIM: usize> {
    /// Vertices stored in a Rc so that they can be accessed from multiple locations.
    /// Mutation after creation is not supported.
    pub vertices: Rc<[na::SVector<f64, DIM>]>,
    /// Storage for each dimension of simplex in the mesh.
    pub(crate) simplices: Vec<SimplexCollection<DIM>>,
    bounds: BoundingBox<DIM>,
}

/// An axis-aligned bounding box.
#[derive(Clone, Copy, Debug)]
pub struct BoundingBox<const DIM: usize> {
    /// The minimum (bottom left in 2D) corner of the box.
    pub min: na::SVector<f64, DIM>,
    /// The maximum (top right in 2D) corner of the box.
    pub max: na::SVector<f64, DIM>,
}

#[derive(Clone, Debug)]
pub(crate) struct SimplexCollection<const MESH_DIM: usize> {
    /// points per simplex in the storage Vec
    simplex_size: usize,
    /// indices stored in a flat Vec to avoid generics for dimension
    pub indices: Vec<usize>,
    /// matrix where the rows correspond to DIM-simplices,
    /// the columns to (DIM-1) simplices,
    /// and the values of -1 or 1 to the relative orientation of the boundary.
    /// this matrix is the coboundary operator for (DIM-1) simplices,
    /// but it's stored with DIM-simplices
    /// because its rows can be efficiently used to navigate to boundary simplices.
    boundary_map: nas::CsrMatrix<i8>,
    /// transpose of the DIM+1-dimensional collection's `boundary_map`,
    /// stored separately for efficient access.
    /// the rows in this correspond to DIM-simplices again,
    /// and the columns to DIM+1-simplices.
    coboundary_map: nas::CsrMatrix<i8>,
    /// simplices on the boundary of the mesh.
    mesh_boundary: fb::FixedBitSet,
    /// circumcenters and barycenters Rc'd so that 0-simplices
    /// can have the mesh vertices here without duplicating data
    circumcenters: Rc<[na::SVector<f64, MESH_DIM>]>,
    barycenters: Rc<[na::SVector<f64, MESH_DIM>]>,
    /// barycentric differentials are not always needed
    /// and take a fair bit of memory (`simplex_size` N-vectors per simplex),
    /// so they're computed lazily on first access
    barycentric_differentials: OnceCell<Vec<na::SVector<f64, MESH_DIM>>>,
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
            barycenters: Rc::from([]),
            barycentric_differentials: OnceCell::new(),
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

    /// Get the slice of vertex indices corresponding to a single simplex.
    fn simplex_indices(&self, simplex_idx: usize) -> &[usize] {
        let start_idx = simplex_idx * self.simplex_size;
        &self.indices[start_idx..start_idx + self.simplex_size]
    }
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

    #[inline]
    pub fn get_simplex_by_index<const DIM: usize>(
        &self,
        idx: usize,
    ) -> SimplexView<'_, DIM, MESH_DIM> {
        let start_idx = idx * self.simplices[DIM].simplex_size;
        let idx_range = start_idx..start_idx + self.simplices[DIM].simplex_size;
        SimplexView {
            vertices: &self.vertices,
            indices: &self.simplices[DIM].indices[idx_range],
        }
    }

    pub fn simplices<const DIM: usize>(&self) -> SimplexIter<'_, DIM, MESH_DIM> {
        SimplexIter {
            mesh: self,
            index: 0,
            len: self.simplices[DIM].len(),
        }
    }

    /// Access the vertex indices for the given dimension of simplex
    /// as a chunked iterator where each element is a `DIM + 1`-length slice
    /// containing the indices of one simplex.
    #[inline]
    pub fn indices<const DIM: usize>(&self) -> std::slice::ChunksExact<'_, usize>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        self.simplices[DIM].indices.chunks_exact(DIM + 1)
    }

    /// Get a slice of `DIM`-simplex barycenters.
    #[inline]
    pub fn barycenters<const DIM: usize>(&self) -> &[na::SVector<f64, MESH_DIM>]
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        &self.simplices[DIM].barycenters
    }

    /// Get a bounding box enclosing the entire mesh.
    #[inline]
    pub fn bounds(&self) -> BoundingBox<MESH_DIM> {
        self.bounds
    }

    /// Create a new cochain with a value of zero
    /// for each `DIM`-simplex in the mesh.
    ///
    /// See the [crate-level docs][crate#operators] for usage information.
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

    /// Create a cochain with values supplied by a function
    /// that takes the vertices of a cell and produces a scalar.
    ///
    /// This does not integrate a function automatically;
    /// the user is expected to compute the integral over a simplex themselves.
    /// There are plans to implement quadratures for numerical integration
    /// that can be used with this in `dexterior`, but for now it must be done by hand.
    ///
    /// For primal simplices, the number of vertices in the slice
    /// given to the integration function is guaranteed to be `DIM + 1`.
    /// For dual cells of dimension 2 and above, the number can vary.
    ///
    /// NOTE: The only cases of this that are fully implemented
    /// are primal 0- and 1-simplices and dual 0-cells.
    /// There is an initial sketch of an implementation for the general case,
    /// but it currently gives vertices in an arbitrary order
    /// that may not match the orientation of the cell.
    /// It also hasn't been tested so correctness isn't guaranteed.
    /// Use with extreme caution.
    pub fn integrate_cochain<const DIM: usize, Primality, IntgFn>(
        &self,
        mut integrate: IntgFn,
    ) -> crate::Cochain<DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
        IntgFn: FnMut(&[na::SVector<f64, MESH_DIM>]) -> f64,
    {
        let mut c = self.new_zero_cochain::<DIM, Primality>();

        match (DIM, Primality::IS_PRIMAL) {
            // special cases for 0-simplices since they don't need to be collected into a slice
            (0, true) => {
                for (vertex, c_val) in izip!(self.vertices.iter(), c.values.iter_mut()) {
                    *c_val = integrate(&[*vertex]);
                }
            }
            (0, false) => {
                // dual 0-simplices are the circumcenters of highest-dimensional simplices
                for (circumcenter, c_val) in izip!(
                    self.simplices[MESH_DIM].circumcenters.iter(),
                    c.values.iter_mut()
                ) {
                    *c_val = integrate(&[*circumcenter]);
                }
            }
            // higher-dimensional simplices
            // (incomplete cases!! see doc comment)
            (_, true) => {
                // buffer to hold vertices of the current cell
                // so that we can pass them into the function as a slice
                // (their placement in the source data is generally not contiguous)
                let mut vertices = Vec::new();

                for (simplex, c_val) in izip!(self.simplices::<DIM>(), c.values.iter_mut()) {
                    vertices.extend(simplex.vertices());
                    *c_val = integrate(&vertices);
                    vertices.clear();
                }
            }
            (_, false) => {
                let mut vertices = Vec::new();
                let primal_dim = MESH_DIM - DIM;
                // the dual cell is composed of circumcenters of all coboundary simplices
                for (simplex_idx, (cob_row, c_val)) in izip!(
                    self.simplices[primal_dim].coboundary_map.row_iter(),
                    c.values.iter_mut()
                )
                .enumerate()
                {
                    for &cob_idx in cob_row.col_indices() {
                        let cob_circumcenter =
                            self.simplices[primal_dim + 1].circumcenters[cob_idx];
                        vertices.push(cob_circumcenter);
                    }
                    // boundary dual 1-simplices are a special case
                    // as they end on the boundary `MESH_DIM - 1`-simplex
                    // instead of connecting two `MESH_DIM`-simplices like the rest
                    if DIM == 1 && vertices.len() <= 1 {
                        vertices.push(self.simplices[primal_dim].circumcenters[simplex_idx])
                    }

                    *c_val = integrate(&vertices);
                    vertices.clear();
                }
            }
        }

        c
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
            |(primal_vol, dual_vol): (&f64, &f64)| *dual_vol / *primal_vol
        } else {
            |(primal_vol, dual_vol): (&f64, &f64)| *primal_vol / *dual_vol
        };

        let diag = na::DVector::from_iterator(
            simplex_count,
            izip!(
                &self.simplices[primal_dim].volumes,
                &self.simplices[primal_dim].dual_volumes
            )
            .map(compute_diag_val),
        );

        // for dual star, compute sign to match the definition
        // star^2 = (-1)^(k*(n-k))
        // where n is the mesh dimension and k is the primal star dimension
        let diag = if !Primality::IS_PRIMAL && primal_dim * (MESH_DIM - primal_dim) % 2 != 0 {
            -diag
        } else {
            diag
        };

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

    /// Find the index of a simplex in its collection given its vertex indices.
    ///
    /// This is not useful very often, but can be used
    /// to associate a cochain value with a simplex
    /// in cases when the simplex index is not known but its vertices are.
    ///
    /// Returns None if no `DIM`-simplex with the given indices exists.
    /// This is always the case if the number of indices isn't `DIM + 1`.
    pub fn find_simplex_index<const DIM: usize>(&self, indices: &[usize]) -> Option<usize>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        let simplices = &self.simplices[DIM];
        // construct a slice of slices containing each simplex's indices,
        // so that we can use the built-in slice::binary_search function.
        // we can do this because the simplices are sorted lexicographically.
        //
        // it would be more efficient in large meshes to build a hashmap
        // from index arrays to simplex indices for this,
        // but that is annoying to do in a dimension-generic way.
        // try to do it if this ends up being used in a performance-critical place
        let indices_chunked: Vec<&[usize]> = simplices
            .indices
            .chunks_exact(simplices.simplex_size)
            .collect();

        indices_chunked.binary_search(&indices).ok()
    }

    /// Get an iterator over barycentric differentials of each `DIM`-simplex.
    /// This is primarily useful for constructing Whitney forms by hand.
    ///
    /// The returned iterator yields slices of `DIM + 1` vectors.
    /// This method isn't implemented for 0-simplices.
    pub fn barycentric_differentials<const DIM: usize>(
        &self,
    ) -> std::slice::ChunksExact<na::SVector<f64, MESH_DIM>>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        na::Const<DIM>: na::DimNameSub<na::U1>,
    {
        let simplices = &self.simplices[DIM];
        let bary_diffs = simplices.barycentric_differentials.get_or_init(|| {
            mesh_construction::compute_barycentric_differentials::<DIM, MESH_DIM>(self)
        });
        bary_diffs.chunks_exact(simplices.simplex_size)
    }

    /// Interpolate a 1-cochain defined on the edges of this mesh
    /// into vectors at simplex barycenters using the Whitney map.
    ///
    /// This method is only defined for 2- and higher-dimensional meshes.
    pub fn barycentric_interpolate_1(
        &self,
        c: &crate::Cochain<1, Primal>,
    ) -> Vec<na::SVector<f64, MESH_DIM>>
    where
        na::Const<MESH_DIM>:
            na::DimNameSub<na::U1> + na::DimNameSub<na::U2> + na::DimNameSub<na::Const<MESH_DIM>>,
    {
        let mut whitney_vals = Vec::new();

        let top_simplices = &self.simplices[MESH_DIM];
        for (indices, bary_diffs) in izip!(
            top_simplices
                .indices
                .chunks_exact(top_simplices.simplex_size),
            self.barycentric_differentials::<MESH_DIM>(),
        ) {
            let mut whitney_val = na::SVector::zeros();
            // each combination of two vertices in the simplex
            // is one of the 1-simplex edges contributing to the Whitney map
            for (start, end) in
                (0..indices.len() - 1).flat_map(|s| (s + 1..indices.len()).map(move |e| (s, e)))
            {
                let edge_indices = [indices[start], indices[end]];
                // there's room for optimization here;
                // find_simplex_index runs a binary search every time.
                // if this is a notable performance hit,
                // try recursively going through boundary maps instead
                let cochain_idx = self.find_simplex_index::<1>(&edge_indices).unwrap();
                // the barycentric coordinates at the barycenter are all 1 / MESH_DIM.
                // we'll multiply this into the value at the end
                whitney_val += c.values[cochain_idx] * (bary_diffs[start] - bary_diffs[end]);
            }
            whitney_val /= MESH_DIM as f64;
            whitney_vals.push(whitney_val);
        }

        whitney_vals
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
// (circumcenter, volume etc.) as well as boundaries and coboundaries through these views

pub struct SimplexView<'a, const DIM: usize, const MESH_DIM: usize> {
    indices: &'a [usize],
    // view into all vertices of the mesh,
    // indexed into by values in the `indices` slice
    vertices: &'a [na::SVector<f64, MESH_DIM>],
}

impl<'a, const DIM: usize, const MESH_DIM: usize> SimplexView<'a, DIM, MESH_DIM> {
    pub fn vertices(&self) -> impl '_ + Iterator<Item = na::SVector<f64, MESH_DIM>> {
        self.indices.iter().map(|i| self.vertices[*i])
    }
}

pub struct SimplexIter<'a, const DIM: usize, const MESH_DIM: usize> {
    mesh: &'a SimplicialMesh<MESH_DIM>,
    index: usize,
    len: usize,
}

impl<'a, const DIM: usize, const MESH_DIM: usize> Iterator for SimplexIter<'a, DIM, MESH_DIM> {
    type Item = SimplexView<'a, DIM, MESH_DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        if self.index >= self.len {
            return None;
        }
        Some(self.mesh.get_simplex_by_index::<DIM>(self.index))
    }
}
