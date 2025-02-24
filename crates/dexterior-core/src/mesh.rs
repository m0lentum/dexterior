//! The core discretization structure of DEC, the (simplicial) mesh.

/// Low-level mesh construction and corresponding tests.
mod mesh_construction;
/// re-export the testing meshes for use in other modules' tests
/// (pub because they're used outside of the core crate)
#[doc(hidden)]
pub use mesh_construction::{tiny_mesh_2d, tiny_mesh_3d};

/// Views and iterators over simplices and dual cells.
mod views;
use views::IndexIter;
pub use views::{DualCellIter, DualCellView, SimplexIter, SimplexView};

mod subset;
pub use subset::{Subset, SubsetImpl};

//

use fixedbitset as fb;
use nalgebra as na;
use nalgebra_sparse as nas;

use itertools::izip;
use std::{cell::OnceCell, collections::HashMap, rc::Rc};

use crate::{
    cochain::{Cochain, CochainImpl},
    operator::Scaling,
    quadrature::Quadrature,
};

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
    /// map from the vertex indices of a simplex to its index in this collection.
    /// constructed lazily in `SimplicialMesh::find_simplex_index`.
    index_map: OnceCell<HashMap<Vec<usize>, usize>>,
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
    /// user-defined subsets, e.g. physical groups from gmsh meshes.
    custom_subsets: HashMap<String, fb::FixedBitSet>,
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
    /// only for MESH_DIM-simplices:
    /// orientation of the simplex with sorted indices
    /// relative to the canonical orientation
    /// (lower-dimensional simplices' orientation is only meaningful in the context of boundaries,
    /// given by the boundary maps)
    orientations: Vec<i8>,
}

impl<const DIM: usize> Default for SimplexCollection<DIM> {
    fn default() -> Self {
        // initialize empty collections
        // which will be filled during mesh construction
        Self {
            simplex_size: 0,
            indices: Vec::new(),
            index_map: OnceCell::new(),
            boundary_map: nas::CsrMatrix::zeros(0, 0),
            coboundary_map: nas::CsrMatrix::zeros(0, 0),
            mesh_boundary: fb::FixedBitSet::default(),
            custom_subsets: HashMap::new(),
            circumcenters: Rc::from([]),
            barycenters: Rc::from([]),
            barycentric_differentials: OnceCell::new(),
            volumes: Vec::new(),
            dual_volumes: Vec::new(),
            orientations: Vec::new(),
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

    /// Get a view into a simplex by its index in the data.
    #[inline]
    pub fn get_simplex_by_index<const DIM: usize>(
        &self,
        idx: usize,
    ) -> SimplexView<'_, na::Const<DIM>, MESH_DIM>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        self.get_simplex_by_index_impl::<na::Const<DIM>>(idx)
    }

    fn get_simplex_by_index_impl<Dim: na::DimName>(
        &self,
        idx: usize,
    ) -> SimplexView<'_, Dim, MESH_DIM> {
        let start_idx = idx * self.simplices[Dim::USIZE].simplex_size;
        let idx_range = start_idx..start_idx + self.simplices[Dim::USIZE].simplex_size;
        SimplexView {
            mesh: self,
            index: idx,
            indices: &self.simplices[Dim::USIZE].indices[idx_range],
            _marker: std::marker::PhantomData,
        }
    }

    /// Get a slice of all vertices in the mesh.
    #[inline]
    pub fn vertices(&self) -> &[na::SVector<f64, MESH_DIM>] {
        &self.vertices
    }

    /// Iterate over all `DIM`-dimensional simplices in the mesh.
    pub fn simplices<const DIM: usize>(&self) -> SimplexIter<'_, DIM, MESH_DIM>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        SimplexIter {
            mesh: self,
            idx_iter: IndexIter::All(0..self.simplices[DIM].len()),
        }
    }

    /// Iterate over the `DIM`-simplices in the given subset.
    pub fn simplices_in<'me, 'sub: 'me, const DIM: usize>(
        &'me self,
        subset: &'sub Subset<DIM, Primal>,
    ) -> SimplexIter<'me, DIM, MESH_DIM>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        SimplexIter {
            mesh: self,
            idx_iter: IndexIter::Subset(subset.indices.ones()),
        }
    }

    /// Iterate over all `DIM`-dimensional dual cells in the mesh.
    pub fn dual_cells<const DIM: usize>(&self) -> DualCellIter<'_, DIM, MESH_DIM>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        DualCellIter {
            mesh: self,
            idx_iter: IndexIter::All(0..self.simplices[MESH_DIM - DIM].len()),
        }
    }

    /// Iterate over the `DIM`-dimensional dual cells in the given subset.
    pub fn dual_cells_in<'me, 'sub: 'me, const DIM: usize>(
        &'me self,
        subset: &'sub Subset<DIM, Dual>,
    ) -> DualCellIter<'me, DIM, MESH_DIM>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        DualCellIter {
            mesh: self,
            idx_iter: IndexIter::Subset(subset.indices.ones()),
        }
    }

    /// Get a subset representing the boundary of a subset of `MESH_DIM`-simplices.
    ///
    /// These are the `MESH_DIM - 1`-simplices
    /// whose coboundary only includes one of this subset's simplices.
    pub fn subset_boundary(
        &self,
        subset: &Subset<MESH_DIM, Primal>,
    ) -> SubsetImpl<na::DimNameDiff<na::Const<MESH_DIM>, na::U1>, Primal>
    where
        // funky trait bounds to get the iterators in the impl working,
        // these shouldn't limit anything in practice
        na::Const<MESH_DIM>: na::DimNameSub<na::U1>
            + na::DimNameSub<na::Const<MESH_DIM>>
            + na::DimNameSub<na::DimNameSum<na::DimNameDiff<na::Const<MESH_DIM>, na::U1>, na::U1>>
            + na::DimNameSub<na::DimNameDiff<na::Const<MESH_DIM>, na::U1>>,
        na::DimNameDiff<na::Const<MESH_DIM>, na::U1>: na::DimNameAdd<na::U1>,
    {
        let simplices = self
            .simplices_in::<MESH_DIM>(subset)
            .flat_map(|s| s.boundary().map(|(_, b)| b))
            .filter(|b| {
                b.coboundary()
                    .filter(|(_, cob)| subset.indices.contains(cob.index()))
                    .count()
                    == 1
            });
        SubsetImpl::<na::DimNameDiff<na::Const<MESH_DIM>, na::U1>, Primal>::from_simplex_iter(
            simplices,
        )
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

    /// Create a cochain by integrating a function over cells of the mesh.
    ///
    /// See the [`quadrature`][crate::quadrature] module for available options.
    ///
    /// NOTE: The only cases of this that are fully implemented
    /// are primal 0- and 1-simplices and dual 0-cells.
    /// There is an initial sketch of an implementation for the general case,
    /// but it currently gives vertices in an arbitrary order
    /// that may not match the orientation of the cell.
    /// It also hasn't been tested so correctness isn't guaranteed.
    /// Use with extreme caution.
    pub fn integrate_cochain<const DIM: usize, Primality>(
        &self,
        quadrature: impl Quadrature<DIM, MESH_DIM>,
    ) -> crate::Cochain<DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        let cell_count = if Primality::IS_PRIMAL {
            self.simplex_count_dyn(DIM)
        } else {
            self.simplex_count_dyn(MESH_DIM - DIM)
        };
        self.integrate_cochain_impl(IndexIter::All(0..cell_count), quadrature)
    }

    /// Create a cochain by integrating over cells,
    /// only performing the integration for the given subset.
    ///
    /// This currently has some severe limitations in its usefulness.
    /// See [`integrate_cochain`][Self::integrate_cochain] for details.
    pub fn integrate_cochain_partial<const DIM: usize, Primality>(
        &self,
        subset: &Subset<DIM, Primality>,
        quadrature: impl Quadrature<DIM, MESH_DIM>,
    ) -> crate::Cochain<DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        self.integrate_cochain_impl(IndexIter::Subset(subset.indices.ones()), quadrature)
    }

    /// Overwrite a subset of an existing cochain with newly integrated values.
    ///
    /// Convenience method combining [`integrate_cochain_partial`][Self::integrate_cochain_partial]
    /// and [`Cochain::overwrite`].
    /// Has the same limitations as the other `integrate_cochain` methods.
    pub fn integrate_overwrite<const DIM: usize, Primality>(
        &self,
        cochain: &mut Cochain<DIM, Primality>,
        subset: &Subset<DIM, Primality>,
        quadrature: impl Quadrature<DIM, MESH_DIM>,
    ) where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>> + Copy,
        Primality: MeshPrimality + Copy,
    {
        cochain.overwrite(subset, &self.integrate_cochain_partial(subset, quadrature));
    }

    fn integrate_cochain_impl<const DIM: usize, Primality>(
        &self,
        idx_iter: IndexIter,
        quadrature: impl Quadrature<DIM, MESH_DIM>,
    ) -> crate::Cochain<DIM, Primality>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
        Primality: MeshPrimality,
    {
        let mut c = self.new_zero_cochain::<DIM, Primality>();

        match (DIM, Primality::IS_PRIMAL) {
            // special cases for 0-simplices since they don't need to be collected into a slice
            (0, true) => {
                for idx in idx_iter {
                    c.values[idx] = quadrature.compute(&[self.vertices[idx]]);
                }
            }
            (0, false) => {
                // dual 0-simplices are the circumcenters of highest-dimensional simplices
                for idx in idx_iter {
                    c.values[idx] =
                        quadrature.compute(&[self.simplices[MESH_DIM].circumcenters[idx]]);
                }
            }
            // higher-dimensional simplices
            // (incomplete cases!! see doc comment)
            (_, true) => {
                // buffer to hold vertices of the current cell
                // so that we can pass them into the function as a slice
                // (their placement in the source data is generally not contiguous)
                let mut vertices = Vec::new();

                for idx in idx_iter {
                    let simplex = self.get_simplex_by_index::<DIM>(idx);
                    vertices.extend(simplex.vertices());
                    c.values[idx] = quadrature.compute(&vertices);
                    vertices.clear();
                }
            }
            (_, false) => {
                let mut vertices = Vec::new();
                let primal_dim = MESH_DIM - DIM;
                // the dual cell is composed of circumcenters of all coboundary simplices
                // TODO: this is not true -
                // dual cell of a k-simplex is bounded by the circumcenters of (N-k)-simplices
                for idx in idx_iter {
                    let cob_row = self.simplices[primal_dim].coboundary_map.row(idx);
                    for &cob_idx in cob_row.col_indices() {
                        let cob_circumcenter =
                            self.simplices[primal_dim + 1].circumcenters[cob_idx];
                        vertices.push(cob_circumcenter);
                    }
                    // boundary dual 1-simplices are a special case
                    // as they end on the boundary `MESH_DIM - 1`-simplex
                    // instead of connecting two `MESH_DIM`-simplices like the rest
                    if DIM == 1 && vertices.len() <= 1 {
                        vertices.push(self.simplices[primal_dim].circumcenters[idx])
                    }

                    c.values[idx] = quadrature.compute(&vertices);
                    vertices.clear();
                }
            }
        }

        c
    }

    /// Construct an exterior derivative operator.
    ///
    /// See the [crate-level docs][crate#operators] for usage information.
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
    /// See the [crate-level docs][crate#operators] for usage information.
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

    /// Create a nonuniform scaling matrix that assigns coefficients to elements
    /// of a primal cochain according to the given function.
    ///
    /// Due to type system constraints, this method only applies to primal cochains.
    /// Use [`scaling_dual`][Self::scaling_dual] instead if scaling a dual cochain.
    pub fn scaling<const DIM: usize>(
        &self,
        elem_scaling: impl Fn(SimplexView<na::Const<DIM>, MESH_DIM>) -> f64,
    ) -> Scaling<DIM, Primal>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        let diag = na::DVector::from_iterator(
            self.simplex_count::<DIM>(),
            self.simplices::<DIM>().map(elem_scaling),
        );
        Scaling::new(diag)
    }

    /// Create a nonuniform scaling matrix that assigns coefficients to elements
    /// of a dual cochain according to the given function.
    ///
    /// Due to type system constraints, this method only applies to dual cochains.
    /// Use [`scaling`][Self::scaling] instead if scaling a primal cochain.
    pub fn scaling_dual<const DIM: usize>(
        &self,
        elem_scaling: impl Fn(DualCellView<na::Const<DIM>, MESH_DIM>) -> f64,
    ) -> Scaling<DIM, Dual>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        let diag = na::DVector::from_iterator(
            self.simplex_count_dyn(MESH_DIM - DIM),
            self.dual_cells::<DIM>().map(elem_scaling),
        );
        Scaling::new(diag)
    }

    /// Compute a discrete wedge product between two primal cochains.
    ///
    /// The output is a primal cochain whose dimension
    /// is the sum of the input cochains' dimensions.
    /// This method is only defined if the output dimension
    /// is less than or equal to the mesh dimension.
    ///
    /// There are several possible ways to define a discrete wedge product.
    /// We use the metric-independent definition given on page 74, eq. 7.2.1
    /// of Hirani (2003). Discrete Exterior Calculus.
    pub fn wedge<const D1: usize, const D2: usize>(
        &self,
        c1: &Cochain<D1, Primal>,
        c2: &Cochain<D2, Primal>,
    ) -> CochainImpl<na::DimNameSum<na::Const<D1>, na::Const<D2>>, Primal>
    where
        na::Const<D1>: na::DimNameAdd<na::Const<D2>>,
        na::Const<MESH_DIM>: na::DimNameSub<na::DimNameSum<na::Const<D1>, na::Const<D2>>>,
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<D1>>,
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<D2>>,
    {
        let ret_dim: usize = D1 + D2;
        let ret_simplex_count = self.simplex_count_dyn(ret_dim);
        let mut ret = CochainImpl::zeros(ret_simplex_count);

        let permutations = crate::permutation::Permutations::new(ret_dim + 1);
        // scaling factor 1/(k+l+1)! in the formula
        let scaling = 1. / (1..=(D1 + D2 + 1)).product::<usize>() as f64;

        // buffers to hold boundary simplices' vertex indices,
        // needed because we must sort them to easily find the corresponding simplex
        let mut left_indices: Vec<usize> = Vec::with_capacity(D1 + 1);
        let mut right_indices: Vec<usize> = Vec::with_capacity(D2 + 1);

        // neat bit of type inference for the dimension of simplex
        // from the fact that we use the simplex to assign to `ret`
        for simplex in (0..ret_simplex_count).map(|i| self.get_simplex_by_index_impl(i)) {
            // if the output is a top-dimensional simplex,
            // we also need to consider the orientation of the simplex
            // because vertices are in lexicographic order, not consistently oriented
            // (lower-dimensional simplices on the other hand are consistently oriented)
            let simplex_orientation = if ret_dim < MESH_DIM {
                1.
            } else {
                self.simplices[MESH_DIM].orientations[simplex.index()] as f64
            };
            let mut sum = 0.;
            for perm in permutations.iter() {
                // we need to find the k- and l-dimensional boundary simplices
                // corresponding to each permutation
                // as well as their orientation relative to the permutation
                // to get the right values out of the cochains

                left_indices.extend(perm.indices[..=D1].iter().map(|i| simplex.indices[*i]));
                right_indices.extend(perm.indices[D1..].iter().map(|i| simplex.indices[*i]));
                let left_parity = crate::permutation::get_parity(&left_indices) as f64;
                let right_parity = crate::permutation::get_parity(&right_indices) as f64;

                // sort the indices to find the corresponding simplices using find_simplex_index,
                // and thus the cochain values we need.
                // this works because all simplices have their indices in ascending order.
                // we could also sidestep the hashmap lookup in find_simplex_index
                // by recursively iterating through boundaries
                // but this seems easier and likely similar in terms of performance
                left_indices.sort();
                right_indices.sort();
                let left_simplex = self
                    .find_simplex_index_impl::<na::Const<D1>>(&left_indices)
                    .unwrap();
                let right_simplex = self
                    .find_simplex_index_impl::<na::Const<D2>>(&right_indices)
                    .unwrap();

                sum += perm.sign as f64
                    * simplex_orientation
                    * (left_parity
                        * c1.values[left_simplex]
                        * right_parity
                        * c2.values[right_simplex]);

                left_indices.clear();
                right_indices.clear();
            }
            ret[simplex] = scaling * sum;
        }

        ret
    }

    /// Get the set of `DIM`-simplices on the mesh boundary.
    ///
    /// This method does not exist for the highest-dimensional simplices in the mesh,
    /// as they are all in the mesh interior.
    #[inline]
    pub fn boundary<const DIM: usize>(&self) -> Subset<DIM, Primal>
    where
        na::Const<DIM>: na::DimNameAdd<na::U1>,
        na::Const<MESH_DIM>: na::DimNameSub<na::DimNameSum<na::Const<DIM>, na::U1>>,
    {
        Subset::new(self.simplices[DIM].mesh_boundary.clone())
    }

    /// Look up a subset of `DIM`-simplices, defined as a `gmsh` physical group, by name.
    ///
    /// Returns None if a subset with the name does not exist for this dimension.
    ///
    /// Note that the mesh boundary is a special subset
    /// accessed through [`boundary`][Self::boundary] instead of this method.
    #[inline]
    pub fn get_subset<const DIM: usize>(&self, name: &str) -> Option<Subset<DIM, Primal>>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        let indices = self.simplices[DIM].custom_subsets.get(name)?.clone();
        Some(Subset::<DIM, Primal>::new(indices))
    }

    /// Store a named subset in the mesh data.
    /// Not exposed to the user; used internally by the `gmsh` module.
    #[inline]
    pub(crate) fn store_subset<const DIM: usize>(
        &mut self,
        name: &str,
        subset: Subset<DIM, Primal>,
    ) {
        self.simplices[DIM]
            .custom_subsets
            .insert(name.to_string(), subset.indices);
    }

    /// Find the index of a simplex in its collection given its vertex indices.
    ///
    /// This is not useful very often, but can be used
    /// to associate a cochain value with a simplex
    /// in cases when the simplex index is not known but its vertices are.
    ///
    /// Returns None if no `DIM`-simplex with the given indices exists.
    /// This is always the case if the number of indices isn't `DIM + 1`.
    #[inline]
    pub fn find_simplex_index<const DIM: usize>(&self, indices: &[usize]) -> Option<usize>
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        self.find_simplex_index_impl(indices)
    }

    fn find_simplex_index_impl<Dim>(&self, indices: &[usize]) -> Option<usize>
    where
        Dim: na::DimName,
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        // special case for 0-simplices since this is just the vertex index directly
        if Dim::USIZE == 0 {
            return Some(indices[0]);
        }

        let simplices = &self.simplices[Dim::USIZE];
        let index_map = simplices.index_map.get_or_init(|| {
            simplices
                .indices
                .chunks_exact(simplices.simplex_size)
                .enumerate()
                .map(|(i, vert_is)| (Vec::from(vert_is), i))
                .collect()
        });
        index_map.get(indices).copied()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that the discrete wedge product gives expected results.
    #[test]
    fn wedge() {
        let mesh_2d = tiny_mesh_2d();

        // check with cochains where each value
        // is an integer equal to the simplex index
        let mut c0 = mesh_2d.new_zero_cochain::<0, Primal>();
        let mut c1 = mesh_2d.new_zero_cochain::<1, Primal>();
        // need a second 1-cochain different from the first
        // because a cochain wedged with itself is always zero
        let mut c1_2 = mesh_2d.new_zero_cochain::<1, Primal>();
        let mut c2 = mesh_2d.new_zero_cochain::<2, Primal>();

        for vals in [
            c0.values.iter_mut(),
            c1.values.iter_mut(),
            c2.values.iter_mut(),
        ] {
            for (i, v) in vals.enumerate() {
                *v = i as f64;
            }
        }
        for (i, v) in c1_2.values.iter_mut().rev().enumerate() {
            *v = i as f64;
        }

        let wedge_00 = mesh_2d.wedge(&c0, &c0);
        let expected_00: Cochain<0, Primal> =
            Cochain::from_values(vec![0., 1., 4., 9., 16., 25., 36.].into());
        assert_eq!(
            wedge_00, expected_00,
            "0-cochains should be pointwise multiplied by wedge"
        );

        let wedge_01 = mesh_2d.wedge(&c0, &c1);
        let wedge_10 = mesh_2d.wedge(&c1, &c0);
        assert_eq!(wedge_01, wedge_10, "wedge with 0-cochain should commute");
        let expected_01: Cochain<1, Primal> = Cochain::from_values(
            vec![0., 1., 3., 6., 10., 12.5, 21., 24.5, 32., 40.5, 50., 60.5].into(),
        );
        assert_eq!(wedge_01, expected_01, "wrong result from 0-1 wedge");

        let wedge_11_self = mesh_2d.wedge(&c1, &c1);
        let expected_11: Cochain<2, Primal> = mesh_2d.new_zero_cochain();
        assert_eq!(
            wedge_11_self, expected_11,
            "wedge of 1-cochain with itself should be zero"
        );

        let wedge_11_flipped = mesh_2d.wedge(&c1_2, &c1);
        let wedge_11 = mesh_2d.wedge(&c1, &c1_2);
        assert_eq!(
            wedge_11, -wedge_11_flipped,
            "wedge of 1-cochains should anticommute"
        );
        let expected_11: Cochain<2, Primal> =
            Cochain::from_values(vec![-14. - 2. / 3., 11., -14. - 2. / 3., 11., -11., 11.].into());
        assert_eq!(wedge_11, expected_11, "wrong result from 1-1 wedge");

        // just going to trust that the correctness of these tests
        // translates to further cases in 3d and beyond
        // because computing these by hand is insanely cumbersome
    }
}
