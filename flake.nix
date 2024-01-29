{
  description = "Pytket Quantinuum Extension";
  inputs.nixpkgs.url = "github:nixos/nixpkgs";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  inputs.tket.url = "github:CQCL/tket";
  inputs.tket.inputs.nixpkgs.follows = "nixpkgs";

  inputs.pytket-qir.url = "github:CQCL/pytket-qir/feature/nix-support";
  inputs.pytket-qir.inputs.nixpkgs.follows = "nixpkgs";
  inputs.pytket-qir.inputs.tket.follows = "tket";

  outputs = { self, nixpkgs, flake-utils, tket, pytket-qir }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (self: super: {
              inherit (tket.packages."${system}") tket pytket;
              inherit (pytket-qir.packages."${system}") pytket-qir;
            })
            (import ./nix-support/pytket-quantinuum.nix)
          ];
        };
      in {
        packages = {
          pytket-quantinuum = pkgs.pytket-quantinuum;
        };
        devShells = {
          default = pkgs.mkShell { buildInputs = [ pkgs.pytket-quantinuum ]; };
        };
        checks = {
          pytket-quantinuum-tests = pkgs.pytket-quantinuum;
        };
      });
}
