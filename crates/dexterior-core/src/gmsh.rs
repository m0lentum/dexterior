//! Utilities for loading meshes generated with [`gmsh`](https://www.gmsh.info/).
//!
//! Only version 4.1 of the MSH format is supported,
//! as per the [`mshio`] library.

use std::collections::{HashMap, HashSet};

use crate::mesh::Subset;

/// Error in loading a mesh from a Gmsh .msh file.
#[derive(thiserror::Error, Debug)]
pub enum GmshError {
    /// Error parsing the .msh file.
    ///
    /// (Implementation note: parser error converted to string
    /// to avoid lifetime issues with the byte slices it contains)
    #[error("Parsing the .msh data failed")]
    ParseError(String),
    /// The given .msh file contains no nodes.
    #[error("Invalid .msh data: no nodes")]
    MissingNodes,
    /// The given .msh file contains no elements of the supported type.
    #[error("Invalid .msh data: no elements of the correct type")]
    MissingElements,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct EntityId {
    dim: i32,
    tag: i32,
}

#[derive(Clone, Debug, Default)]
struct PhysicalGroup {
    entities: HashSet<EntityId>,
    nodes: HashSet<usize>,
}

/// Load a 2D triangle mesh from a `.msh` file.
///
/// First-order triangle elements in the file are interpreted as the triangles of the mesh.
/// These must be of type `Tri3` (see [`ElementType`][mshio::ElementType]).
/// The `z` coordinate of vertices is dropped to project the mesh to 2D space.
///
/// ```
/// # use dexterior_core::gmsh::{load_trimesh_2d, GmshError};
/// # fn load() -> Result<(), Box<dyn std::error::Error>> {
/// # let msh_path = "../dexterior/examples/meshes/2d_square_pi_x_pi.msh";
/// let msh_bytes = std::fs::read(msh_path)?;
/// let mesh = load_trimesh_2d(&msh_bytes)?;
/// # Ok(())
/// # }
/// # load().expect("Failed to load mesh");
/// ```
///
/// # Physical groups
///
/// If the .msh file contains physical groups,
/// mesh subsets of all dimensions are generated corresponding to them.
/// These subsets can be looked up with the physical group's integer tag
/// using [`get_subset`][crate::SimplicialMesh::get_subset].
/// (named groups are currently not supported due to limitations of [`mshio`]).
///
/// Notably, these subsets only include simplices
/// where **every** vertex belongs to the group.
/// Consequently, any group containing entities with dimension higher than 0
/// (curves, surfaces, etc.)
/// must also contain its boundary entities.
/// For example, in order for a 1-simplex boundary curve
/// to contain all its line segments,
/// the end points of each curve must belong
/// to the same physical group as the curves themselves.
/// Otherwise the line segments ending at those points are not included.
///
/// For example, to generate a triangle-shaped mesh
/// with a physical group containing the bottom boundary edge,
/// you could use the following geometry:
/// ```text
/// Point(1) = {-1, 0, 0, 0.1};
/// Point(2) = {1, 0, 0, 0.1};
/// Point(3) = {0, 1, 0, 0.1};
/// Line(1) = {1, 2};
/// Line(2) = {2, 3};
/// Line(3) = {3, 1};
/// Curve Loop(1) = {1, 2, 3};
/// Plane Surface(1) = {1};
///
/// Physical Curve(100) = {1};
/// Physical Point(100) = {1, 2}; // also add the curve endpoints!
/// ```
/// Then
/// ```
/// # use dexterior_core::mesh::tiny_mesh_2d;
/// # let mesh = tiny_mesh_2d();
/// mesh.get_subset::<0>("100"); // vertices
/// mesh.get_subset::<1>("100"); // line segments
/// ```
/// would return the group in question.
/// (note: gmsh does not save elements that aren't part of a physical group by default,
/// so remember to use `-save_all` if you copy this code)
pub fn load_trimesh_2d(bytes: &[u8]) -> Result<crate::SimplicialMesh<2>, GmshError> {
    let msh = mshio::parse_msh_bytes(bytes).map_err(|e| GmshError::ParseError(format!("{}", e)))?;
    let nodes = msh.data.nodes.ok_or(GmshError::MissingNodes)?;
    let elements = msh.data.elements.ok_or(GmshError::MissingElements)?;

    let mut physical_groups = gather_physical_groups(msh.data.entities.as_ref());

    let mut vertices: Vec<nalgebra::SVector<f64, 2>> = Vec::new();
    for block in &nodes.node_blocks {
        let ent_id = EntityId {
            dim: block.entity_dim,
            tag: block.entity_tag,
        };
        let mut phys_groups: Vec<&mut PhysicalGroup> = physical_groups
            .values_mut()
            .filter(|pg| pg.entities.contains(&ent_id))
            .collect();

        for node in &block.nodes {
            let vert_idx = vertices.len();
            vertices.push(nalgebra::SVector::<f64, 2>::new(node.x, node.y));
            for g in &mut phys_groups {
                g.nodes.insert(vert_idx);
            }
        }
    }

    if vertices.is_empty() {
        return Err(GmshError::MissingNodes);
    }

    let indices: Vec<usize> = elements
        .element_blocks
        .iter()
        .filter(|block| block.element_type == mshio::ElementType::Tri3)
        .flat_map(|block| block.elements.iter())
        .flat_map(|el| el.nodes.iter())
        // gmsh tags start at 1, subtract 1 to get the index in the array.
        // (this assumes tags are sequential and in order!
        // empirically this seems to be true even in the presence of sparse ids,
        // as these only affect entities)
        .map(|node_tag| *node_tag as usize - 1)
        .collect();
    if indices.is_empty() {
        return Err(GmshError::MissingElements);
    }

    // generate subsets from the vertices' physical group associations

    let mut mesh = crate::SimplicialMesh::new(vertices, indices);
    for (group_id, group) in physical_groups.iter() {
        let group_name = format!("{}", group_id);
        // the subset of vertices is just the nodes in the group
        let verts = Subset::from_indices(group.nodes.iter().cloned());
        mesh.store_subset::<0>(&group_name, verts);
        // for other dimensions, only include simplices for which
        // all vertices belong to the group
        let edges = Subset::from_predicate(&mesh, |s| {
            s.vertex_indices().all(|i| group.nodes.contains(&i))
        });
        mesh.store_subset::<1>(&group_name, edges);

        let tris = Subset::from_predicate(&mesh, |s| {
            s.vertex_indices().all(|i| group.nodes.contains(&i))
        });
        mesh.store_subset::<2>(&group_name, tris);
    }

    Ok(mesh)
}

/// Load a 3D tetrahedral mesh from a `.msh` file.
///
/// First-order tetrahedron elements in the file are interpreted as the tetrahedra of the mesh.
/// These must be of type `Tet4` (see [`ElementType`][mshio::ElementType]).
///
/// See [`load_trimesh_2d`] for information about physical groups.
pub fn load_tetmesh_3d(bytes: &[u8]) -> Result<crate::SimplicialMesh<3>, GmshError> {
    let msh = mshio::parse_msh_bytes(bytes).map_err(|e| GmshError::ParseError(format!("{}", e)))?;
    let nodes = msh.data.nodes.ok_or(GmshError::MissingNodes)?;
    let elements = msh.data.elements.ok_or(GmshError::MissingElements)?;

    let mut physical_groups = gather_physical_groups(msh.data.entities.as_ref());

    let mut vertices: Vec<nalgebra::SVector<f64, 3>> = Vec::new();
    for block in &nodes.node_blocks {
        let ent_id = EntityId {
            dim: block.entity_dim,
            tag: block.entity_tag,
        };
        let mut phys_groups: Vec<&mut PhysicalGroup> = physical_groups
            .values_mut()
            .filter(|pg| pg.entities.contains(&ent_id))
            .collect();

        for node in &block.nodes {
            let vert_idx = vertices.len();
            vertices.push(nalgebra::SVector::<f64, 3>::new(node.x, node.y, node.z));
            for g in &mut phys_groups {
                g.nodes.insert(vert_idx);
            }
        }
    }

    if vertices.is_empty() {
        return Err(GmshError::MissingNodes);
    }

    let indices: Vec<usize> = elements
        .element_blocks
        .iter()
        .filter(|block| block.element_type == mshio::ElementType::Tet4)
        .flat_map(|block| block.elements.iter())
        .flat_map(|el| el.nodes.iter())
        .map(|node_tag| *node_tag as usize - 1)
        .collect();
    if indices.is_empty() {
        return Err(GmshError::MissingElements);
    }

    let mut mesh = crate::SimplicialMesh::new(vertices, indices);
    for (group_id, group) in physical_groups.iter() {
        let group_name = format!("{}", group_id);
        let verts = Subset::from_indices(group.nodes.iter().cloned());
        mesh.store_subset::<0>(&group_name, verts);

        let edges = Subset::from_predicate(&mesh, |s| {
            s.vertex_indices().all(|i| group.nodes.contains(&i))
        });
        mesh.store_subset::<1>(&group_name, edges);

        let tris = Subset::from_predicate(&mesh, |s| {
            s.vertex_indices().all(|i| group.nodes.contains(&i))
        });
        mesh.store_subset::<2>(&group_name, tris);

        let tets = Subset::from_predicate(&mesh, |s| {
            s.vertex_indices().all(|i| group.nodes.contains(&i))
        });
        mesh.store_subset::<3>(&group_name, tets);
    }

    Ok(mesh)
}

/// Collect the physical groups defined in a .msh file
/// into a structure we can easily look them up from.
///
/// This only populates the `entities` field of each group;
/// nodes need to be filled in by each individual loader method.
fn gather_physical_groups(
    entities: Option<&mshio::Entities<i32, f64>>,
) -> HashMap<i32, PhysicalGroup> {
    let Some(entities) = entities else {
        return HashMap::new();
    };

    let mut groups: HashMap<i32, PhysicalGroup> = HashMap::new();

    for point in &entities.points {
        for ptag in &point.physical_tags {
            let group = groups.entry(*ptag).or_default();
            group.entities.insert(EntityId {
                dim: 0,
                tag: point.tag,
            });
        }
    }

    for curve in &entities.curves {
        for ptag in &curve.physical_tags {
            let group = groups.entry(*ptag).or_default();
            group.entities.insert(EntityId {
                dim: 1,
                tag: curve.tag,
            });
        }
    }

    for surface in &entities.surfaces {
        for ptag in &surface.physical_tags {
            let group = groups.entry(*ptag).or_default();
            group.entities.insert(EntityId {
                dim: 2,
                tag: surface.tag,
            });
        }
    }

    for volume in &entities.volumes {
        for ptag in &volume.physical_tags {
            let group = groups.entry(*ptag).or_default();
            group.entities.insert(EntityId {
                dim: 3,
                tag: volume.tag,
            });
        }
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_groups_2d() {
        let mesh =
            load_trimesh_2d(include_bytes!("gmsh/test_mesh_2d.msh")).expect("Failed to load mesh");

        // mesh contains one physical group for the bottom edge,
        // ensure it's there and contains the entire edge
        let subset_0 = mesh.get_subset::<0>("100").expect("subset didn't exist");
        let subset_1 = mesh.get_subset::<1>("100").expect("subset didn't exist");
        let subset_2 = mesh.get_subset::<2>("100").expect("subset didn't exist");

        assert!(
            !subset_0.indices.is_empty(),
            "subset should contain vertices"
        );
        assert!(!subset_1.indices.is_empty(), "subset should contain edges");
        assert!(
            subset_2.indices.is_empty(),
            "subset shouldn't contain faces"
        );

        // test that the subsets only contain bottom edge vertices
        // (TODO: subset iterators that return simplices directly)
        for simp in subset_0
            .indices
            .ones()
            .map(|i| mesh.get_simplex_by_index::<0>(i))
        {
            for vert in simp.vertices() {
                assert_eq!(
                    vert.y, 0.,
                    "contained a vertex {vert:?} that wasn't on the bottom edge"
                );
            }
        }

        for simp in subset_1
            .indices
            .ones()
            .map(|i| mesh.get_simplex_by_index::<1>(i))
        {
            for vert in simp.vertices() {
                assert_eq!(
                    vert.y, 0.,
                    "contained an edge {vert:?} that wasn't on the bottom edge"
                );
            }
        }

        // test that all bottom edges and vertices are in the subsets

        for (vert_idx, vert) in mesh
            .vertices()
            .iter()
            .enumerate()
            .filter(|(_, v)| v.y == 0.)
        {
            assert!(
                subset_0.indices.contains(vert_idx),
                "a bottom vertex {vert:?} was not in the subset"
            );
        }

        for (edge_idx, edge) in mesh
            .simplices::<1>()
            .enumerate()
            .filter(|(_, e)| e.vertices().all(|v| v.y == 0.))
        {
            assert!(
                subset_1.indices.contains(edge_idx),
                "a bottom edge {edge:?} was not in the subset"
            );
        }
    }

    #[test]
    fn physical_groups_3d() {
        let mesh =
            load_tetmesh_3d(include_bytes!("gmsh/test_mesh_3d.msh")).expect("Failed to load mesh");

        // mesh contains one physical group for the bottom face (z = 0),
        // ensure it's there and contains the entire face
        let subset_0 = mesh.get_subset::<0>("100").expect("subset didn't exist");
        let subset_1 = mesh.get_subset::<1>("100").expect("subset didn't exist");
        let subset_2 = mesh.get_subset::<2>("100").expect("subset didn't exist");
        let subset_3 = mesh.get_subset::<3>("100").expect("subset didn't exist");

        assert!(
            !subset_0.indices.is_empty(),
            "subset should contain vertices"
        );
        assert!(!subset_1.indices.is_empty(), "subset should contain edges");
        assert!(!subset_2.indices.is_empty(), "subset should contain faces");
        assert!(
            subset_3.indices.is_empty(),
            "subset shouldn't contain volumes"
        );

        // test that the subsets only contain bottom edge vertices
        // (TODO: subset iterators that return simplices directly)
        for simp in subset_0
            .indices
            .ones()
            .map(|i| mesh.get_simplex_by_index::<0>(i))
        {
            for vert in simp.vertices() {
                assert_eq!(
                    vert.z, 0.,
                    "contained a vertex {vert:?} that wasn't on the bottom face"
                );
            }
        }

        for simp in subset_1
            .indices
            .ones()
            .map(|i| mesh.get_simplex_by_index::<1>(i))
        {
            for vert in simp.vertices() {
                assert_eq!(
                    vert.z, 0.,
                    "contained an edge {vert:?} that wasn't on the bottom face"
                );
            }
        }

        for simp in subset_2
            .indices
            .ones()
            .map(|i| mesh.get_simplex_by_index::<2>(i))
        {
            for vert in simp.vertices() {
                assert_eq!(
                    vert.z, 0.,
                    "contained a face {vert:?} that wasn't on the bottom face"
                );
            }
        }

        // test that all bottom edges and vertices are in the subsets

        for (vert_idx, vert) in mesh
            .vertices()
            .iter()
            .enumerate()
            .filter(|(_, v)| v.z == 0.)
        {
            assert!(
                subset_0.indices.contains(vert_idx),
                "a bottom vertex {vert:?} was not in the subset"
            );
        }

        for (edge_idx, edge) in mesh
            .simplices::<1>()
            .enumerate()
            .filter(|(_, e)| e.vertices().all(|v| v.z == 0.))
        {
            assert!(
                subset_1.indices.contains(edge_idx),
                "a bottom edge {edge:?} was not in the subset"
            );
        }

        for (face_idx, face) in mesh
            .simplices::<2>()
            .enumerate()
            .filter(|(_, e)| e.vertices().all(|v| v.z == 0.))
        {
            assert!(
                subset_2.indices.contains(face_idx),
                "a bottom face {face:?} was not in the subset"
            );
        }
    }
}
