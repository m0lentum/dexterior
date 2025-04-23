# Changelog

## 0.3.0 - 2025-04-23

### Features

Core:
- discrete wedge product
- nonuniform scaling operator
- `SimplicialMesh::integrate_overwrite`
- new subset API where subsets are owned by the user
- arithmetic operations for subsets (union, intersection, difference, complement)
- construct operators directly from matrices
- implement traits that make simplex & cell views easier to use (Copy, Eq)
- dual-primal interpolation for 0-cochains on 2D meshes (experimental)

Visuals:
- text rendering
- wireframes for subsets of edges
- user-configurable keyboard interactions

### Fixes

Visuals:
- correct type of input cochain in `Painter::vertex_colors_dual`
- draw meshes with more than u16::MAX vertices

## 0.2.0 - 2024-10-02

### Features

Core:
- polymorphic API for chochain integration; numerical quadratures for line integrals
- ability to restrict integration to a subset of a cochain
- user-defined subsets
  - can be defined manually or mapped from `gmsh` physical groups
- iterators over the simplices belonging to a subset
- access to more information (boundaries, volumes etc.) via simplex views
- index into cochains with SimplexView
- dual cell views and iterators
- arithmetic ops for more permutations of referred/owned cochains

Visuals:
- support WASM/WebGL
- visualize the dual mesh

### Fixes

Core:
- Restrict all mesh methods' dimension to the dimension of the mesh

## 0.1.1 - 2024-01-30

Added README.md to the main crate's Cargo package.

## 0.1.0 - 2024-01-30

Initial release.
