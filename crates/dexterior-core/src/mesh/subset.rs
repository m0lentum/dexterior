use fixedbitset as fb;
use nalgebra as na;

use super::{Dual, DualCellView, MeshPrimality, Primal, SimplexView, SimplicialMesh};

/// A subset of cells in a mesh.
///
/// Can be used to restrict operations to certain parts of the mesh,
/// e.g. with [`MatrixOperator::exclude_subset`
/// ][crate::MatrixOperator::exclude_subset],
/// useful for boundary conditions and spatially varying parameters.
///
/// You can iterate over the cells in this set with [`SimplicialMesh::simplices_in`]
/// or [`SimplicialMesh::dual_cells_in`].
pub type Subset<const DIM: usize, Primality> = SubsetImpl<na::Const<DIM>, Primality>;

/// The subset type used internally by dexterior,
/// with type generics enabling compile-time arithmetic for dimension.
/// Prefer the more convenient type alias [`Subset`] in public APIs.
#[derive(Clone, Debug)]
pub struct SubsetImpl<Dimension, Primality> {
    /// A bitset containing the indices of simplices present in the subset.
    ///
    /// Iterate over the indices with `indices.ones()`.
    pub indices: fb::FixedBitSet,
    _marker: std::marker::PhantomData<(Dimension, Primality)>,
}

impl<Dim, Primality> PartialEq for SubsetImpl<Dim, Primality>
where
    Dim: na::DimName,
{
    fn eq(&self, other: &Self) -> bool {
        self.indices.eq(&other.indices)
    }
}
impl<Dim, Primality> Eq for SubsetImpl<Dim, Primality> where Dim: na::DimName {}

impl<Dim, Primality> SubsetImpl<Dim, Primality>
where
    Dim: na::DimName,
    Primality: MeshPrimality,
{
    #[inline]
    pub(super) fn new(indices: fb::FixedBitSet) -> Self {
        Self {
            indices,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a subset from an iterator of simplex indices.
    pub fn from_indices(indices: impl IntoIterator<Item = usize>) -> Self {
        let bits = fb::FixedBitSet::from_iter(indices);
        Self::new(bits)
    }

    /// Create an empty subset.
    ///
    /// This isn't particularly useful on its own,
    /// but can be handy when taking unions of many subsets.
    pub fn new_empty() -> Self {
        Self::new(fb::FixedBitSet::new())
    }

    /// Create a subset containing every `DIM`-cell in the mesh.
    ///
    /// This isn't particularly useful on its own,
    /// but can be handy when taking intersections of many subsets.
    pub fn new_full<const MESH_DIM: usize>(mesh: &SimplicialMesh<MESH_DIM>) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        let simplex_count = if Primality::IS_PRIMAL {
            mesh.simplex_count_dyn(Dim::USIZE)
        } else {
            mesh.simplex_count_dyn(MESH_DIM - Dim::USIZE)
        };
        let mut indices = fb::FixedBitSet::with_capacity(simplex_count);
        indices.set_range(.., true);
        Self::new(indices)
    }

    /// Take the complement of a subset, i.e. the cells not in that subset.
    pub fn complement<const MESH_DIM: usize>(&self, mesh: &SimplicialMesh<MESH_DIM>) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        let mut indices = Self::new_full(mesh).indices;
        indices.set_range(.., true);
        indices.difference_with(&self.indices);
        Self::new(indices)
    }

    /// Create a subset containing the dual elements of this one.
    #[inline]
    pub fn dual<const MESH_DIM: usize>(
        &self,
        _mesh: &SimplicialMesh<MESH_DIM>,
    ) -> SubsetImpl<na::DimNameDiff<na::Const<MESH_DIM>, Dim>, Primality::Opposite>
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        SubsetImpl::new(self.indices.clone())
    }

    /// Take the intersection (i.e. set of cells that are in both)
    /// of this subset with another of the same type.
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        // fixedbitset's in-place operations are probably more efficient
        // than collecting a new bitset from an iterator
        let mut indices = self.indices.clone();
        indices.intersect_with(&other.indices);
        Self::new(indices)
    }

    /// Take the union (i.e. set of cells that are in one or the other)
    /// of this subset with another of the same type.
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        let mut indices = self.indices.clone();
        indices.union_with(&other.indices);
        Self::new(indices)
    }

    /// Take the difference (i.e. set of cells that are in `self` but not in `other`)
    /// of this subset with another of the same type.
    #[inline]
    pub fn difference(&self, other: &Self) -> Self {
        let mut indices = self.indices.clone();
        indices.difference_with(&other.indices);
        Self::new(indices)
    }

    /// Get the number of cells in this subset.
    #[inline]
    pub fn count(&self) -> usize {
        self.indices.count_ones(..)
    }

    /// Apply the boundary operator to this subset,
    /// yielding the set of `k - 1`-cells
    /// which are on the boundary of some cell in this subset.
    ///
    /// This is equivalent to creating subsets with iterators like this:
    /// ```
    /// # use dexterior_core::{Dual, Primal, Subset, mesh::tiny_mesh_2d};
    /// # let mesh = tiny_mesh_2d();
    /// let primal_tris: Subset<2, Primal> = Subset::from_indices([0, 1, 2]);
    /// let boundary_edges: Subset<1, Primal> = primal_tris.boundary(&mesh);
    /// let iter_edges = Subset::from_simplex_iter(
    ///     mesh.simplices_in(&primal_tris).flat_map(|s| s.boundary().map(|(_, bs)| bs))
    /// );
    /// assert_eq!(boundary_edges, iter_edges);
    /// #
    /// # let boundary_verts: Subset<0, Primal> = boundary_edges.boundary(&mesh);
    /// # let iter_verts = Subset::from_simplex_iter(
    /// #    mesh.simplices_in(&boundary_edges).flat_map(|s| s.boundary().map(|(_, bs)| bs))
    /// # );
    /// # assert_eq!(boundary_verts, iter_verts);
    ///
    /// // also works for dual cells
    /// // (these currently have a limited iterator API,
    /// // so the iterator version is even more verbose)
    /// let dual_cells: Subset<2, Dual> = Subset::from_indices([0, 1, 2]);
    /// let boundary_edges: Subset<1, Dual> = dual_cells.boundary(&mesh);
    /// let iter_edges = Subset::from_cell_iter(
    ///     mesh.dual_cells_in(&dual_cells).flat_map(|c| {
    ///         c.dual().coboundary().map(|(_, dbs)| dbs.dual())
    ///     })
    /// );
    /// assert_eq!(boundary_edges, iter_edges);
    /// #
    /// # let boundary_verts: Subset<0, Dual> = boundary_edges.boundary(&mesh);
    /// # let iter_verts = Subset::from_cell_iter(
    /// #    mesh.dual_cells_in(&boundary_edges).flat_map(|c| {
    /// #        c.dual().coboundary().map(|(_, dbs)| dbs.dual())
    /// #    })
    /// # );
    /// # assert_eq!(boundary_verts, iter_verts);
    /// ```
    ///
    /// This method does not exist for 0-cells.
    pub fn boundary<const MESH_DIM: usize>(
        &self,
        mesh: &SimplicialMesh<MESH_DIM>,
    ) -> SubsetImpl<na::DimNameDiff<Dim, na::U1>, Primality>
    where
        Dim: na::DimNameSub<na::U1>,
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        let map = if Primality::IS_PRIMAL {
            &mesh.simplices[Dim::USIZE].boundary_map
        } else {
            &mesh.simplices[MESH_DIM - Dim::USIZE].coboundary_map
        };

        let mut new_indices = fb::FixedBitSet::new();
        for row_idx in self.indices.ones() {
            let Some(row) = map.get_row(row_idx) else {
                continue;
            };
            new_indices.extend(row.col_indices().iter().cloned());
        }
        SubsetImpl::new(new_indices)
    }

    /// Apply the coboundary operator to this subset,
    /// yielding the set of `k + 1`-cells
    /// which have some cell from this subset on their boundary.
    ///
    /// This is equivalent to creating subsets with iterators like this:
    /// ```
    /// # use dexterior_core::{Dual, Primal, Subset, mesh::tiny_mesh_2d};
    /// # let mesh = tiny_mesh_2d();
    /// let primal_verts: Subset<0, Primal> = Subset::from_indices([0, 1, 2]);
    /// let cob_edges: Subset<1, Primal> = primal_verts.coboundary(&mesh);
    /// let iter_edges = Subset::from_simplex_iter(
    ///     mesh.simplices_in(&primal_verts).flat_map(|s| s.coboundary().map(|(_, bs)| bs))
    /// );
    /// assert_eq!(cob_edges, iter_edges);
    ///
    /// # let cob_tris: Subset<2, Primal> = cob_edges.coboundary(&mesh);
    /// # let iter_tris = // ...
    /// # Subset::from_simplex_iter(
    /// #    mesh.simplices_in(&cob_edges).flat_map(|s| s.coboundary().map(|(_, bs)| bs))
    /// # );
    /// # assert_eq!(cob_tris, iter_tris);
    /// #
    /// let dual_verts: Subset<0, Dual> = Subset::from_indices([0, 1, 2]);
    /// let cob_edges: Subset<1, Dual> = dual_verts.coboundary(&mesh);
    /// let iter_edges = Subset::from_cell_iter(
    ///     mesh.dual_cells_in(&dual_verts).flat_map(|c| {
    ///         c.dual().boundary().map(|(_, dbs)| dbs.dual())
    ///     })
    /// );
    /// assert_eq!(cob_edges, iter_edges);
    /// # let cob_tris: Subset<2, Dual> = cob_edges.coboundary(&mesh);
    /// # let iter_tris = // ...
    /// # Subset::from_cell_iter(
    /// #    mesh.dual_cells_in(&cob_edges).flat_map(|c| {
    /// #        c.dual().boundary().map(|(_, dbs)| dbs.dual())
    /// #    })
    /// # );
    /// # assert_eq!(cob_tris, iter_tris);
    /// ```
    ///
    /// This method does not exist for `MESH_DIM`-cells.
    pub fn coboundary<const MESH_DIM: usize>(
        &self,
        mesh: &SimplicialMesh<MESH_DIM>,
    ) -> SubsetImpl<na::DimNameSum<Dim, na::U1>, Primality>
    where
        Dim: na::DimNameAdd<na::U1>,
        na::Const<MESH_DIM>: na::DimNameSub<na::DimNameSum<Dim, na::U1>>,
    {
        let map = if Primality::IS_PRIMAL {
            &mesh.simplices[Dim::USIZE].coboundary_map
        } else {
            &mesh.simplices[MESH_DIM - Dim::USIZE].boundary_map
        };

        let mut new_indices = fb::FixedBitSet::new();
        for row_idx in self.indices.ones() {
            let Some(row) = map.get_row(row_idx) else {
                continue;
            };
            new_indices.extend(row.col_indices().iter().cloned());
        }
        SubsetImpl::new(new_indices)
    }
}

impl<Dim> SubsetImpl<Dim, Primal>
where
    Dim: na::DimName,
{
    /// Create a subset containing the simplices in the given mesh
    /// that pass the given predicate.
    pub fn from_predicate<const MESH_DIM: usize>(
        mesh: &SimplicialMesh<MESH_DIM>,
        pred: impl Fn(SimplexView<Dim, MESH_DIM>) -> bool,
    ) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        let bits: fb::FixedBitSet = (0..mesh.simplex_count_dyn(Dim::USIZE))
            .map(|i| mesh.get_simplex_by_index_impl(i))
            .enumerate()
            .filter(|(_, s)| pred(*s))
            .map(|(i, _)| i)
            .collect();
        Self::new(bits)
    }

    /// Create a subset containing the simplices yielded by an iterator.
    pub fn from_simplex_iter<'a, const MESH_DIM: usize>(
        iter: impl Iterator<Item = SimplexView<'a, Dim, MESH_DIM>>,
    ) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        let bits: fb::FixedBitSet = iter.map(|s| s.index()).collect();
        Self::new(bits)
    }

    /// Check if the subset contains a given simplex.
    #[inline]
    pub fn contains<const MESH_DIM: usize>(&self, simplex: SimplexView<'_, Dim, MESH_DIM>) -> bool
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        self.indices.contains(simplex.index())
    }
}

impl<Dim> SubsetImpl<Dim, Dual>
where
    Dim: na::DimName,
{
    /// Create a subset containing the dual cells in the mesh
    /// that pass the given predicate.
    pub fn from_predicate_dual<const MESH_DIM: usize>(
        mesh: &SimplicialMesh<MESH_DIM>,
        pred: impl Fn(DualCellView<Dim, MESH_DIM>) -> bool,
    ) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        let bits: fb::FixedBitSet = (0..mesh.dual_cell_count_dyn(Dim::USIZE))
            .map(|i| mesh.get_dual_cell_by_index_impl::<Dim>(i))
            .enumerate()
            .filter(|(_, c)| pred(*c))
            .map(|(i, _)| i)
            .collect();
        Self::new(bits)
    }

    /// Create a subset containing the dual cells yielded by an iterator.
    pub fn from_cell_iter<'a, const MESH_DIM: usize>(
        iter: impl Iterator<Item = DualCellView<'a, Dim, MESH_DIM>>,
    ) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        let bits: fb::FixedBitSet = iter.map(|s| s.index()).collect();
        Self::new(bits)
    }

    /// Check if the subset contains a given dual cell.
    #[inline]
    pub fn contains<const MESH_DIM: usize>(&self, cell: DualCellView<'_, Dim, MESH_DIM>) -> bool
    where
        na::Const<MESH_DIM>: na::DimNameSub<Dim>,
    {
        self.indices.contains(cell.index())
    }
}

impl<const MESH_DIM: usize> SubsetImpl<na::Const<MESH_DIM>, Primal> {
    /// Get a subset representing the boundary manifold of a region
    /// defined by a subset of highest-dimensional simplices in a mesh.
    ///
    /// These are the `N - 1`-simplices
    /// whose coboundary only includes one of this subset's simplices.
    /// This is different from [`boundary`][Self::boundary],
    /// which applies the boundary operator to a subset,
    /// yielding every `k - 1`-cell that bounds any cell in that subset.
    pub fn manifold_boundary(
        &self,
        mesh: &SimplicialMesh<MESH_DIM>,
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
        let simplices = mesh
            .simplices_in::<MESH_DIM>(self)
            .flat_map(|s| s.boundary().map(|(_, b)| b))
            .filter(|b| {
                b.coboundary()
                    .filter(|(_, cob)| self.indices.contains(cob.index()))
                    .count()
                    == 1
            });
        SubsetImpl::<na::DimNameDiff<na::Const<MESH_DIM>, na::U1>, Primal>::from_simplex_iter(
            simplices,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::mesh_construction::tiny_mesh_3d;

    /// Check that subset creation works as expected.
    #[test]
    fn create_subsets() {
        let mesh = tiny_mesh_3d();

        let indices = [1, 3, 8];
        let idx_subset: Subset<1, Primal> = Subset::from_indices(indices.iter().cloned());
        let pred_subset: Subset<1, Primal> =
            Subset::from_predicate(&mesh, |s| indices.contains(&s.index()));
        let iter_subset = Subset::from_cell_iter(mesh.simplices_in(&pred_subset).map(|s| s.dual()));

        assert_eq!(idx_subset.indices, pred_subset.indices);
        assert_eq!(pred_subset.indices, iter_subset.indices);
        itertools::assert_equal(idx_subset.indices.ones(), indices.iter().cloned());

        let pred_complement = pred_subset.complement(&mesh);
        for edge in mesh.simplices::<1>() {
            assert!(pred_subset.contains(edge) ^ pred_complement.contains(edge));
        }

        let full_subset = Subset::<2, Dual>::new_full(&mesh);
        let empty_subset = Subset::<2, Dual>::new_empty();
        let empty_complement = empty_subset.complement(&mesh);
        for cell in mesh.dual_cells() {
            assert!(full_subset.contains(cell));
            assert!(!empty_subset.contains(cell));
            assert!(empty_complement.contains(cell));
        }
    }
}
