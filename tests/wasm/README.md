Create a C file containing

```c
void init() {}
```

at the top, and any other int-to-int functions below.

Run:

```shell
clang --target=wasm32 -Xclang -target-abi -Xclang experimental-mv --no-standard-libraries -Wl,--export-all -Wl,--no-entry -o <filename>.wasm <filename>.c
```

to generate the wasm file.

You can then run:

```shell
wasm2wat <filename>.wasm -o <filename>.wast
```

to convert the WASM to human-readable text format.
