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

#[derive(Clone, Debug)]
struct SimplexCollection {
    /// points per simplex in the storage Vec
    simplex_size: usize,
    /// indices stored in a flat Vec to avoid generics for dimension
    indices: Vec<usize>,
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
            indices: Vec::new(),
        });

        // highest dimension simplices have given indices
        simplices[DIM - 1].indices = indices;

        // rest of the levels are inferred from boundaries of the top-level simplices
        let mut level_iter = simplices.iter_mut().rev().peekable();
        while let Some(upper_simplices) = level_iter.next() {
            let Some(lower_simplices) = level_iter.peek_mut() else {
                break;
            };

            // buffer to sort the simplex currently being processed.
            // indices for each simplex are sorted in increasing order
            // so that simplices can be unambiguously identified with their set of vertices
            let mut sorted_simplex: Vec<usize> = Vec::with_capacity(lower_simplices.simplex_size);

            for upper_simplex in upper_simplices
                .indices
                .chunks_exact(upper_simplices.simplex_size)
            {
                // every unique combination of vertices in the upper simplex
                // is a simplex on its boundary
                for exclude_idx in upper_simplex {
                    sorted_simplex.clear();
                    sorted_simplex.extend(upper_simplex.iter().filter(|i| *i != exclude_idx));
                    sorted_simplex.sort_unstable();

                    // linear search through already added simplices to deduplicate.
                    // this isn't the most efficient for large meshes,
                    // but we'll optimize later if it becomes a problem
                    if lower_simplices
                        .indices
                        .chunks_exact(lower_simplices.simplex_size)
                        .any(|s| s == sorted_simplex)
                    {
                        continue;
                    }

                    lower_simplices.indices.extend_from_slice(&sorted_simplex);

                    // TODO: orientation
                    // TODO: store boundary information
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

    /// A small hexagon-shaped 2D mesh for testing basic functionality.
    /// Shaped somewhat like this:
    ///    ____
    ///   /\  /\
    ///  /__\/__\
    ///  \  /\  /
    ///   \/__\/
    ///
    /// with vertices and triangles ordered left to right, top to bottom.
    fn simple_mesh_2d() -> SimplicialMesh<2> {
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
            0, 3, 1,
            1, 3, 4,
            2, 5, 3,
            3, 5, 6,
            3, 6, 4,
        ];
        SimplicialMesh::new(vertices, indices)
    }

    /// Lower-dimensional simplices are generated correctly
    /// for a simple 2d mesh.
    #[test]
    fn simple_2d_simplices_are_correct() {
        let mesh = simple_mesh_2d();
        #[rustfmt::skip]
        let expected_1_simplices = vec![
            2, 3,   0, 3,   0, 2,
            1, 3,   0, 1,
            3, 4,   1, 4,
            3, 5,   2, 5,
            5, 6,   3, 6,
            4, 6,
        ];
        assert_eq!(expected_1_simplices, mesh.simplices[0].indices);
    }
}
