use super::SimplicialMesh;

use fixedbitset as fb;
use nalgebra as na;
use nalgebra_sparse as nas;

/// A view into a single simplex's data.
///
/// This type can also be used as an index into a primal [`Cochain`][crate::Cochain]
/// of the corresponding dimension, providing a shorter syntax
/// and a type check to ensure the dimensions match:
/// ```
/// # use dexterior_core::{mesh::tiny_mesh_3d, Cochain, Primal, Dual};
/// # let mesh_3d = tiny_mesh_3d();
/// let c_primal: Cochain<1, Primal> = mesh_3d.new_zero_cochain();
/// let c_dual: Cochain<2, Dual> = mesh_3d.new_zero_cochain();
/// for edge in mesh_3d.simplices::<1>() {
///     let primal_val = c_primal[edge];
///     let dual_val = c_dual[edge.dual()];
///     // ..is a typechecked equivalent to
///     let primal_val = c_primal.values[edge.index()];
///     let dual_val = c_dual.values[edge.index()];
/// }
/// ```
/// Something like this wouldn't compile, for instance:
/// ```compile_fail
/// # use dexterior_core::{mesh::tiny_mesh_3d, Cochain, Dual};
/// # let mesh_3d = tiny_mesh_3d();
/// let c: Cochain<1, Dual> = mesh_3d.new_zero_cochain();
/// for edge in mesh_3d.simplices::<1>() {
///     let val = c[edge];
///     // ..but the index method would still work (almost certainly incorrectly):
///     let val = c.values[edge.index()];
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SimplexView<'a, SimplexDim, const MESH_DIM: usize> {
    pub(super) mesh: &'a SimplicialMesh<MESH_DIM>,
    pub(super) index: usize,
    pub(super) indices: &'a [usize],
    pub(super) _marker: std::marker::PhantomData<SimplexDim>,
}

impl<'a, SimplexDim, const MESH_DIM: usize> PartialEq for SimplexView<'a, SimplexDim, MESH_DIM> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<'a, SimplexDim, const MESH_DIM: usize> Eq for SimplexView<'a, SimplexDim, MESH_DIM> {}

impl<'a, SimplexDim, const MESH_DIM: usize> SimplexView<'a, SimplexDim, MESH_DIM>
where
    SimplexDim: na::DimName,
    na::Const<MESH_DIM>: na::DimNameSub<SimplexDim>,
{
    /// Iterate over the vertices of this simplex.
    #[inline]
    pub fn vertices(&self) -> impl '_ + Iterator<Item = na::SVector<f64, MESH_DIM>> {
        self.indices.iter().map(|i| self.mesh.vertices[*i])
    }

    /// Iterate over the vertex indices of this simplex.
    #[inline]
    pub fn vertex_indices(&self) -> impl '_ + Iterator<Item = usize> {
        self.indices.iter().cloned()
    }

    /// Get the index of this simplex in the ordering of `DIM`-simplices.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the unsigned volume of this simplex.
    #[inline]
    pub fn volume(&self) -> f64 {
        self.mesh.simplices[SimplexDim::USIZE].volumes[self.index]
    }

    /// Get the unsigned volume of the dual cell corresponding to this simplex.
    #[inline]
    pub fn dual_volume(&self) -> f64 {
        self.mesh.simplices[SimplexDim::USIZE].dual_volumes[self.index]
    }

    /// Get the circumcenter of this simplex.
    #[inline]
    pub fn circumcenter(&self) -> na::SVector<f64, MESH_DIM> {
        self.mesh.simplices[SimplexDim::USIZE].circumcenters[self.index]
    }

    /// Get the barycenter of this simplex.
    #[inline]
    pub fn barycenter(&self) -> na::SVector<f64, MESH_DIM> {
        self.mesh.simplices[SimplexDim::USIZE].barycenters[self.index]
    }

    /// Get the dual cell corresponding to this simplex.
    #[inline]
    pub fn dual(
        self,
    ) -> DualCellView<'a, na::DimNameDiff<na::Const<MESH_DIM>, SimplexDim>, MESH_DIM> {
        DualCellView {
            mesh: self.mesh,
            index: self.index,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, SimplexDim, const MESH_DIM: usize> SimplexView<'a, SimplexDim, MESH_DIM>
where
    SimplexDim: na::DimNameAdd<na::U1>,
    na::Const<MESH_DIM>: na::DimNameSub<na::DimNameSum<SimplexDim, na::U1>>,
{
    /// Iterate over the `k+1`-dimensional simplices on whose boundary this simplex lies.
    pub fn coboundary(self) -> BoundaryIter<'a, na::DimNameSum<SimplexDim, na::U1>, MESH_DIM> {
        BoundaryIter {
            mesh: self.mesh,
            index: 0,
            map_row: self.mesh.simplices[SimplexDim::USIZE]
                .coboundary_map
                .row(self.index),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, SimplexDim, const MESH_DIM: usize> SimplexView<'a, SimplexDim, MESH_DIM>
where
    SimplexDim: na::DimNameSub<na::U1>,
{
    /// Iterate over the `k-1`-dimensional simplices on the boundary of this simplex.
    pub fn boundary(self) -> BoundaryIter<'a, na::DimNameDiff<SimplexDim, na::U1>, MESH_DIM> {
        BoundaryIter {
            mesh: self.mesh,
            index: 0,
            map_row: self.mesh.simplices[SimplexDim::USIZE]
                .boundary_map
                .row(self.index),
            _marker: std::marker::PhantomData,
        }
    }
}

/// A view into a dual cell's data.
///
/// This type can also be used to index into a dual [`Cochain`][crate::Cochain]
/// of the corresponding dimension in a type-checked fashion.
/// See [`SimplexView`] for an example.
#[derive(Clone, Copy, Debug)]
pub struct DualCellView<'a, CellDim, const MESH_DIM: usize> {
    pub(super) mesh: &'a SimplicialMesh<MESH_DIM>,
    pub(super) index: usize,
    pub(super) _marker: std::marker::PhantomData<CellDim>,
}

impl<'a, CellDim, const MESH_DIM: usize> PartialEq for DualCellView<'a, CellDim, MESH_DIM> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<'a, CellDim, const MESH_DIM: usize> Eq for DualCellView<'a, CellDim, MESH_DIM> {}

impl<'a, CellDim, const MESH_DIM: usize> DualCellView<'a, CellDim, MESH_DIM>
where
    CellDim: na::DimName,
    na::Const<MESH_DIM>: na::DimNameSub<CellDim>,
{
    /// Get the index of this cell in the ordering of dual `DIM`-cells.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the unsigned volume of this cell.
    #[inline]
    pub fn volume(&self) -> f64 {
        self.mesh.simplices[MESH_DIM - CellDim::USIZE].dual_volumes[self.index]
    }

    /// Get the primal simplex corresponding to this cell.
    #[inline]
    pub fn dual(self) -> SimplexView<'a, na::DimNameDiff<na::Const<MESH_DIM>, CellDim>, MESH_DIM> {
        self.mesh.get_simplex_by_index_impl(self.index)
    }
}

/// An iterator over the boundary or coboundary of a simplex,
/// obtained with [`SimplexView::coboundary`] or [`SimplexView::boundary`].
///
/// The iterator returns a pair `(orientation, simplex)`
/// where orientation is the relative orientation
/// between the boundary and coboundary simplices in question.
pub struct BoundaryIter<'a, SimplexDim, const MESH_DIM: usize> {
    mesh: &'a SimplicialMesh<MESH_DIM>,
    index: usize,
    map_row: nas::csr::CsrRow<'a, i8>,
    _marker: std::marker::PhantomData<SimplexDim>,
}

impl<'a, SimplexDim, const MESH_DIM: usize> Iterator for BoundaryIter<'a, SimplexDim, MESH_DIM>
where
    SimplexDim: na::DimName,
{
    type Item = (i8, SimplexView<'a, SimplexDim, MESH_DIM>);

    fn next(&mut self) -> Option<Self::Item> {
        let cols = self.map_row.col_indices();
        if self.index >= cols.len() {
            return None;
        }
        let next_idx = cols[self.index];
        let next_ori = self.map_row.values()[self.index];

        self.index += 1;

        Some((next_ori, self.mesh.get_simplex_by_index_impl(next_idx)))
    }
}

/// Iterator over a set of `DIM`-simplices in a mesh.
pub struct SimplexIter<'a, const DIM: usize, const MESH_DIM: usize> {
    pub(super) mesh: &'a SimplicialMesh<MESH_DIM>,
    pub(super) idx_iter: IndexIter<'a>,
}

impl<'a, const DIM: usize, const MESH_DIM: usize> Iterator for SimplexIter<'a, DIM, MESH_DIM>
where
    na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
{
    type Item = SimplexView<'a, na::Const<DIM>, MESH_DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_idx = self.idx_iter.next()?;
        Some(self.mesh.get_simplex_by_index::<DIM>(next_idx))
    }
}

/// Iterator over a set of `DIM`-dimensional dual cells in a mesh.
pub struct DualCellIter<'a, const DIM: usize, const MESH_DIM: usize> {
    pub(super) mesh: &'a SimplicialMesh<MESH_DIM>,
    pub(super) idx_iter: IndexIter<'a>,
}

impl<'a, const DIM: usize, const MESH_DIM: usize> Iterator for DualCellIter<'a, DIM, MESH_DIM>
where
    na::Const<MESH_DIM>: na::DimNameSub<na::Const<DIM>>,
{
    type Item = DualCellView<'a, na::Const<DIM>, MESH_DIM>;

    fn next(&mut self) -> Option<Self::Item> {
        let next_idx = self.idx_iter.next()?;
        Some(DualCellView {
            mesh: self.mesh,
            index: next_idx,
            _marker: std::marker::PhantomData,
        })
    }
}

/// A set of indices to iterate over,
/// defined either as a contiguous range
/// or an arbitrary set represented by a bitset.
pub(super) enum IndexIter<'a> {
    All(std::ops::Range<usize>),
    Subset(fb::Ones<'a>),
}

impl<'a> Iterator for IndexIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::All(range) => range.next(),
            Self::Subset(indices) => indices.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn iterators() {
        let mesh = tiny_mesh_2d();
        // for a quick sanity test, check that every edge
        // is on the coboundary of all its boundary simplices and vice versa
        for edge in mesh.simplices::<1>() {
            for (b_ori, b) in edge.boundary() {
                let (c_ori, _c) = b
                    .coboundary()
                    .find(|(_, e)| e.index() == edge.index())
                    .expect("edge must be on its boundary's coboundary");
                assert_eq!(
                    b_ori, c_ori,
                    "boundary and coboundary orientations should agree"
                );
            }

            for (b_ori, b) in edge.coboundary() {
                let (c_ori, _c) = b
                    .boundary()
                    .find(|(_, e)| e.index() == edge.index())
                    .expect("edge must be on its coboundary's boundary");
                assert_eq!(
                    b_ori, c_ori,
                    "boundary and coboundary orientations should agree"
                );
            }
        }

        // ensure simplex views are accessing the correct data

        // volume
        itertools::assert_equal(
            mesh.simplices::<0>().map(|s| s.volume()),
            mesh.simplices[0].volumes.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<1>().map(|s| s.volume()),
            mesh.simplices[1].volumes.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<2>().map(|s| s.volume()),
            mesh.simplices[2].volumes.iter().cloned(),
        );
        // dual volume
        itertools::assert_equal(
            mesh.simplices::<0>().map(|s| s.dual_volume()),
            mesh.simplices[0].dual_volumes.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<1>().map(|s| s.dual_volume()),
            mesh.simplices[1].dual_volumes.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<2>().map(|s| s.dual_volume()),
            mesh.simplices[2].dual_volumes.iter().cloned(),
        );
        // circumcenter
        itertools::assert_equal(
            mesh.simplices::<0>().map(|s| s.circumcenter()),
            mesh.simplices[0].circumcenters.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<1>().map(|s| s.circumcenter()),
            mesh.simplices[1].circumcenters.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<2>().map(|s| s.circumcenter()),
            mesh.simplices[2].circumcenters.iter().cloned(),
        );
        // barycenter
        itertools::assert_equal(
            mesh.simplices::<0>().map(|s| s.barycenter()),
            mesh.simplices[0].barycenters.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<1>().map(|s| s.barycenter()),
            mesh.simplices[1].barycenters.iter().cloned(),
        );
        itertools::assert_equal(
            mesh.simplices::<2>().map(|s| s.barycenter()),
            mesh.simplices[2].barycenters.iter().cloned(),
        );
    }

    #[test]
    fn dual_cell_views() {
        let mesh = tiny_mesh_3d();

        // check that the dual of a dual is the original simplex

        for edge in mesh.simplices::<1>() {
            let dual_face = edge.dual();
            let edge_again = dual_face.dual();
            assert_eq!(edge.index(), edge_again.index());
            itertools::assert_equal(edge.vertices(), edge_again.vertices());
            assert_eq!(edge.dual_volume(), dual_face.volume());
        }

        // check that dual cell and simplex iterators agree

        for (vert, dual_vol) in izip!(mesh.simplices::<0>(), mesh.dual_cells::<3>()) {
            itertools::assert_equal(vert.vertex_indices(), dual_vol.dual().vertex_indices());
        }
        for (edge, dual_face) in izip!(mesh.simplices::<1>(), mesh.dual_cells::<2>()) {
            itertools::assert_equal(edge.vertex_indices(), dual_face.dual().vertex_indices());
        }
        for (face, dual_edge) in izip!(mesh.simplices::<2>(), mesh.dual_cells::<1>()) {
            itertools::assert_equal(face.vertex_indices(), dual_edge.dual().vertex_indices());
        }
        for (vol, dual_vert) in izip!(mesh.simplices::<3>(), mesh.dual_cells::<0>()) {
            itertools::assert_equal(vol.vertex_indices(), dual_vert.dual().vertex_indices());
        }
    }
}
