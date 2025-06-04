{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-25.05";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ (import inputs.rust-overlay) ];
        };

        rust = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" ];
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rust
            pkgs.lld
            pkgs.gmsh
            pkgs.renderdoc
            pkgs.cargo-flamegraph
          ];
          # bunch of dynamically linked libs for gmsh and wgpu
          LD_LIBRARY_PATH = with pkgs.xorg; with pkgs.lib.strings;
            concatStrings (intersperse ":" [
              # gmsh
              "${pkgs.libGLU}/lib"
              "${pkgs.libglvnd}/lib"
              "${pkgs.fontconfig.lib}/lib"
              "${libXrender}/lib"
              "${libXfixes}/lib"
              "${libXft}/lib"
              "${libXinerama}/lib"
              # shared by both
              "${libXcursor}/lib"
              "${libX11}/lib"
              # wgpu
              "${pkgs.libxkbcommon}/lib"
              "${libXxf86vm}/lib"
              "${libXi}/lib"
              "${libXrandr}/lib"
              "${pkgs.vulkan-loader}/lib"
              "${pkgs.stdenv.cc.cc.lib}/lib64"
              "${pkgs.stdenv.cc.cc.lib}/lib64"
            ]);
        };
      });
}
