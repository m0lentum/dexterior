{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-23.05";
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
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rust
            pkgs.lld
            pkgs.gmsh
          ];
        };
        # bunch of dynamically linked libs for gmsh
        LD_LIBRARY_PATH = with pkgs.xorg; with pkgs.lib.strings;
          concatStrings (intersperse ":" [
            "${pkgs.libGLU}/lib"
            "${pkgs.libglvnd}/lib"
            "${pkgs.fontconfig.lib}/lib"
            "${libX11}/lib"
            "${libXrender}/lib"
            "${libXcursor}/lib"
            "${libXfixes}/lib"
            "${libXft}/lib"
            "${libXinerama}/lib"
            "${pkgs.stdenv.cc.cc.lib}/lib64"
          ]);
      });
}
