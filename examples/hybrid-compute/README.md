# Hybrid Compute for QEC Workflows

Quantinuum's Hybrid Compute feature enables users to run classical operations within qubit coherence times. This enables research in quantum error correction and new algorithm designs not otherwise available. This guide covers how to set up and use this feature.

Hybrid Compute is enabled via the use of [Web Assembly (WASM)](https://webassembly.org/), which enables fast, real-time computations.

Wasm itself is a binary instruction format. To write in and compile programs to it, another language such as Rust or C/ C++ is used to write your desired classical functions and compile these to Wasm. The quantum computing workflow will then contain a call to Wasm. This folder contains examples of this workflow. Wasm is used because it is a fast, safe, expressive, and easy to compile to using familiar programming languages. 

## Organization

This subfolder is organized as follows.

---
    ├── Hybrid-compute              <- Examples of how to use Hybrid Compute via WASM
        |
        ├── repeat_until_success    <- Repeat Until Success source code
        ├── repetition_code         <- Repetition Code source code
---

## Setup

To create and use your own Hybrid Compute functions in your workflow, it is necessary to use `Wasm` and a language that compiles code to `Wasm`. These instructions contain the steps for setting up an environment to do this.

It is recommended to either use Rust or C++ for the compilation language to `Wasm`. The languages have functionality that easily compiles to `Wasm`. For other language options see [Wasm Getting Started Developer's page](https://webassembly.org/getting-started/developers-guide/).

### 1. Rust Setup

If you're new to Rust, see the [The Rust Programming Language](https://doc.rust-lang.org/book/title-page.html) book. Note that Cargo is the Rust package manager. For more info on Cargo, see [The Cargo Book](https://doc.rust-lang.org/cargo/guide/).

If using Visual Studio code, you can also install the `rust-analyzer` extension in the Extensions tab.

#### Install Rust

Install Rust using the [Rust Installer](https://www.rust-lang.org/tools/install).

**Windows Users:** Before running the installer, you'll need to install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

#### Create a new Rust project

Navigate to wherever you plan to store your Rust project files, and run: `cargo new --lib rus_wasm`. The project name `rus_wasm` can be whatever project name you want to use.

This creates a new library in a subdirectory with the project name you used with everything you need to get going.

* The `/src/lib.rs` file contains some generated Rust code. This code can be erased and replaced with code you plan to use in your hybrid compute program.
* Inside this subdirectory is a file called `Cargo.toml`. This file will be used to configure your build.

#### Define classical functions in Rust

Define the classical functions you plan to use in your hybrid compute program in the `src/lib.rs` file. Examples of the types of functions that can be put in Rust code are provided in the `Rust` folders within each of the example folders. *We recommend looking at one of the provided examples to know which functions and attributes are required for Hybrid Compute to work.*

#### Compile code to Wasm

1. Configure the `Cargo.toml` file. You can use one of the examples provided to modify yours. 
1. Navigate to your Rust project directory from the command line.
1. Run: `cargo build`. 
    - If successful, a `target` folder and `Cargo.lock` file will be created, alongside the output of the command displaying "Finished."
    - If failed, Cargo will output a messaging related to the reason.
1. Run: `rustup target add wasm32-unknown-unknown`. This allows Rust to used Wasm as a compilation target. This only needs to be run once for your environment. If you have run this previously when compiling Rust to Wasm, you do not need to run it again. 
1. Run: `cargo build --release --target wasm32-unknown-unknown`. This compiles your Rust code to Wasm.

Congratulations! You've succesfully compiled Rust code to Wasm. 

#### Use compiled Wasm

In order to use the compiled wasm from your quantum program, you will want to find the associated `.wasm` file that the compilation step created. This can be found in the `./target/wasm32-unknown-unknown/release` folder of your project. You'll note a file with the name of your project with the extension `.wasm`. To use this in your quantum program, you can copy-paste it into another folder, or ensure that the file path for your Wasm file handler points to this location.

### 2. C/ C++ Setup

#### Install C/ C++

For Windows 10/ 11 users, the Visual Studio build tools (also available through a standard Visual Studio installation) should be downloaded from here. The C++ Desktop Development group should be selected and the clang and LLVM option should be selected.

For Ubuntu 20.04 users, the development tools group needs to be installed with aptitude. The LLVM and clang compilers need to be installed separately.

MacOS users must install XCode command line tools and also brew. The latest version of clang can be installed with brew.

#### Define Classical Functions in C/ C++

Define the classical functions you plan to use in your hybrid compute program in the `src/lib.c` file. Examples of the types of functions that can be put in C code are provided in the `C` folders within each of the example folders. *We recommend looking at one of the provided examples to know which functions and attributes are required for Hybrid Compute to work.*

The standard C (C++) library cannot be used in a program that is intended to be compiled to Wasm.

#### Compilation to Wasm

The command below will compile a *.c source file into a Wasm binary. The Wasm binary, `lib.wasm` will be located in the directory clang is executed. For a *.cpp source file, clang++ should be used intead of clang and lib.c should be replaced by lib.cpp.

```
clang --no-standard-library -Wl,--no-entry -Wl,--export-all -o lib.wasm lib.c
```
