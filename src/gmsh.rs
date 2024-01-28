//! Utilities for loading meshes generated with [`gmsh`](https://www.gmsh.info/).
//!
//! Only version 4.1 of the MSH format is supported,
//! as per the [`mshio`] library.

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

/// Load a 2D triangle mesh from a `.msh` file.
///
/// First-order triangle elements in the file are interpreted as the triangles of the mesh.
/// These must be of type `Tri3` (see [`ElementType`][mshio::ElementType]).
/// The `z` coordinate of vertices is dropped to project the mesh to 2D space.
///
/// ```
/// # use dexterior::gmsh::{load_trimesh_2d, GmshError};
/// # fn load() -> Result<(), Box<dyn std::error::Error>> {
/// let msh_bytes = std::fs::read("examples/meshes/2d_square_pi_x_pi.msh")?;
/// let mesh = load_trimesh_2d(&msh_bytes)?;
/// # Ok(())
/// # }
/// # load().expect("Failed to load mesh");
/// ```
pub fn load_trimesh_2d(bytes: &[u8]) -> Result<crate::SimplicialMesh<2>, GmshError> {
    let msh = mshio::parse_msh_bytes(bytes).map_err(|e| GmshError::ParseError(format!("{}", e)))?;
    let nodes = msh.data.nodes.ok_or(GmshError::MissingNodes)?;
    let elements = msh.data.elements.ok_or(GmshError::MissingElements)?;

    let vertices: Vec<nalgebra::SVector<f64, 2>> = nodes
        .node_blocks
        .iter()
        .flat_map(|block| block.nodes.iter())
        .map(|node| nalgebra::SVector::<f64, 2>::new(node.x, node.y))
        .collect();
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
        // I believe the latter is guaranteed by the format,
        // but the former may not be since tags can be set manually)
        .map(|node_tag| *node_tag as usize - 1)
        .collect();
    if indices.is_empty() {
        return Err(GmshError::MissingElements);
    }

    Ok(crate::SimplicialMesh::new(vertices, indices))
}

/// Load a 3D tetrahedral mesh from a `.msh` file.
///
/// First-order tetrahedron elements in the file are interpreted as the tetrahedra of the mesh.
/// These must be of type `Tet4` (see [`ElementType`][mshio::ElementType]).
pub fn load_tetmesh_3d(bytes: &[u8]) -> Result<crate::SimplicialMesh<3>, GmshError> {
    let msh = mshio::parse_msh_bytes(bytes).map_err(|e| GmshError::ParseError(format!("{}", e)))?;
    let nodes = msh.data.nodes.ok_or(GmshError::MissingNodes)?;
    let elements = msh.data.elements.ok_or(GmshError::MissingElements)?;

    let vertices: Vec<nalgebra::SVector<f64, 3>> = nodes
        .node_blocks
        .iter()
        .flat_map(|block| block.nodes.iter())
        .map(|node| nalgebra::SVector::<f64, 3>::new(node.x, node.y, node.z))
        .collect();
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

    Ok(crate::SimplicialMesh::new(vertices, indices))
}
