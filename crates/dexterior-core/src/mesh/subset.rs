use fixedbitset as fb;
use nalgebra as na;

use super::{Dual, DualCellView, Primal, SimplexView, SimplicialMesh};

/// A subset of cells in a mesh.
///
/// Used to restrict operations to certain parts of the mesh,
/// e.g. with [`ComposedOperator::exclude_subset`
/// ][crate::operator::ComposedOperator::exclude_subset],
/// useful for boundary conditions and spatially varying parameters.
///
/// You can iterate over the simplices in this set with [`SimplicialMesh::simplices_in`].
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

impl<const DIM: usize, Primality> PartialEq for SubsetImpl<na::Const<DIM>, Primality> {
    fn eq(&self, other: &Self) -> bool {
        self.indices.eq(&other.indices)
    }
}
impl<const DIM: usize, Primality> Eq for SubsetImpl<na::Const<DIM>, Primality> {}

impl<const DIM: usize, Primality> SubsetImpl<na::Const<DIM>, Primality> {
    #[inline]
    pub(super) fn new(indices: fb::FixedBitSet) -> Self {
        Self {
            indices,
            _marker: std::marker::PhantomData,
        }
    }

    /// Create a subset from an iterator of simplex indices.
    pub fn from_indices(indices: impl Iterator<Item = usize>) -> Self {
        let bits = fb::FixedBitSet::from_iter(indices);
        Self::new(bits)
    }

    /// Create a subset containing the simplices in the given mesh
    /// that pass the given predicate.
    pub fn from_predicate<const MESH_DIM: usize>(
        mesh: &SimplicialMesh<MESH_DIM>,
        pred: impl Fn(SimplexView<na::Const<DIM>, MESH_DIM>) -> bool,
    ) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        let bits: fb::FixedBitSet = mesh
            .simplices::<DIM>()
            .enumerate()
            .filter(|(_, s)| pred(*s))
            .map(|(i, _)| i)
            .collect();
        Self::new(bits)
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
}

impl<const DIM: usize> SubsetImpl<na::Const<DIM>, Primal>
where
    na::Const<DIM>: na::DimName,
{
    /// Create a subset containing the simplices yielded by an iterator.
    pub fn from_simplex_iter<'a, const MESH_DIM: usize>(
        iter: impl Iterator<Item = SimplexView<'a, na::Const<DIM>, MESH_DIM>>,
    ) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        let bits: fb::FixedBitSet = iter.map(|s| s.index()).collect();
        Self::new(bits)
    }

    /// Check if the subset contains a given simplex.
    #[inline]
    pub fn contains<const MESH_DIM: usize>(
        &self,
        simplex: SimplexView<'_, na::Const<DIM>, MESH_DIM>,
    ) -> bool
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        self.indices.contains(simplex.index())
    }
}

impl<const DIM: usize> SubsetImpl<na::Const<DIM>, Dual>
where
    na::Const<DIM>: na::DimName,
{
    /// Create a subset containing the dual cells yielded by an iterator.
    pub fn from_cell_iter<'a, const MESH_DIM: usize>(
        iter: impl Iterator<Item = DualCellView<'a, na::Const<DIM>, MESH_DIM>>,
    ) -> Self
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        let bits: fb::FixedBitSet = iter.map(|s| s.index()).collect();
        Self::new(bits)
    }

    /// Check if the subset contains a given dual cell.
    #[inline]
    pub fn contains<const MESH_DIM: usize>(
        &self,
        cell: DualCellView<'_, na::Const<DIM>, MESH_DIM>,
    ) -> bool
    where
        na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
    {
        self.indices.contains(cell.index())
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
    }
}
