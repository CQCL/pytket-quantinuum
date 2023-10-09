Create a C file containing

```
void init() {return;}
```

at the top, and any other int-to-int functions below.

Run:

```
clang --target=wasm32 --no-standard-libraries -Wl,--export-all -Wl,--no-entry -o <filename>.wasm <filename>.c
```

to generate the wasm file.

You can then run:

```
wasm2wat <filename>.wasm -o <filename>.wast
```

to convert the WASM to human-readable text format.
