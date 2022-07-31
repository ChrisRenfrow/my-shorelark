let pkgs = import <nixpkgs> { };
in with pkgs;
mkShell {
  nativeBuildInputs = [ openssl.dev ];
  buildInputs = [ gdb udev pkg-config glibc wasm-pack nodejs wabt ];
  shellHook = "";
}
