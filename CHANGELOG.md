# Changelog

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
