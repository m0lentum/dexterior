use nalgebra as na;

/// A mesh composed of only simplices
/// (points, line segments, triangles, tetrahedra etc).
/// This is the primal mesh in a DEC complex.
#[derive(Clone, Debug)]
pub struct SimplicialMesh<const DIM: usize> {
    vertices: Vec<na::SVector<f64, DIM>>,
    /// storage for each dimension of simplex in the mesh
    /// (except 0, as those are just the vertices)
    simplices: [SimplexCollection; DIM],
}

#[derive(Clone, Debug, Default)]
struct SimplexCollection {
    /// points per simplex in the storage Vec
    simplex_size: usize,
    /// indices stored in a flat Vec to avoid generics for dimension
    indices: Vec<usize>,
    /// boundary simplices on the next level down
    boundaries: Vec<BoundarySimplex>,
}

impl SimplexCollection {
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

impl<const DIM: usize> SimplicialMesh<DIM> {
    /// Construct a SimplicialMesh from raw vertices and indices.
    ///
    /// The indices are given as a flat array,
    /// where every `DIM + 1` indices correspond to one `DIM`-simplex.
    pub fn new(vertices: Vec<na::SVector<f64, DIM>>, indices: Vec<usize>) -> Self {
        assert!(DIM > 0, "Cannot create a mesh of dimension 0");

        let mut simplices = std::array::from_fn(|i| SimplexCollection {
            // 0-simplices are omitted from these collections,
            // so the one at index 0 is the collection of 1-simplices,
            // which have 2 points each
            simplex_size: i + 2,
            ..Default::default()
        });

        // highest dimension simplices have the indices given as parameter
        simplices[DIM - 1].indices = indices;
        // by convention, sort simplices to have their indices in ascending order.
        // this simplifies things by enabling a consistent way
        // to identify a simplex with its vertices
        for simplex in simplices[DIM - 1]
            .indices
            .chunks_exact_mut(simplices[DIM - 1].simplex_size)
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
            }
        }

        Self {
            vertices,
            simplices,
        }
    }
}

//
// tests
//

#[cfg(test)]
mod tests {
    use super::*;

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
    fn tiny_mesh_2d() -> SimplicialMesh<2> {
        let vertices = vec![
            Vec2::new(-0.5, 1.0),
            Vec2::new(0.5, 1.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(0.0, 1.0),
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
    fn tiny_mesh_3d() -> SimplicialMesh<3> {
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
            (3, 1), (1, -1), (4, 1),
            (5, 1), (6, -1), (3, 1),
            (7, 1), (8, -1), (0, 1),
            (9, 1), (10, -1), (7, 1),
            (11, 1), (10, -1), (5, 1),
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
            (4, 1), (5, -1), (6, 1), (3, -1),
            (7, 1), (8, -1), (0, 1), (9, -1),
            (10, 1), (11, -1), (4, 1), (9, -1),
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
    }
}
