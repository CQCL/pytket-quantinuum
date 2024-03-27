# Getting Started with Quantinuum's Hybrid Compute Feature

Quantinuum's Hybrid Compute feature enables users to run classical operations within qubit coherence times. This enables research in quantum error correction and new algorithm designs not otherwise available. This guide covers how to set up and use this feature.

Hybrid Compute is enabled via the use of [Web Assembly (Wasm)](https://webassembly.org/), which enables fast, real-time computations.

Wasm itself is a binary instruction format. To write in and compile programs to it, another language such as Rust or C++ is used to write your desired classical functions and compile these to Wasm. The quantum computing workflow will then contain a call to Wasm. This folder contains examples of this workflow. Wasm is used because it is a fast, safe, expressive, and easy to compile to using familiar programming languages. 

## Organization

This subfolder is organized as follows.

---
    ├── wasm                              <- Examples of how to use Hybrid Compute via WASM
          |
          ├── repeat_until_success        <- Repeat Until Success source code
          │
          └── repetition_code             <- Repetition Code source code
---

## Download

If you haven't done so already, download the complete set of examples on the user portal by clicking the **Download** button on the bottom-left of the folder viewer.

## Notebook Example

The python notebook in this directory called `Quantinuum Hybrid Compute via pytket.ipynb` can be run to show an example of the hybrid compute feature's workflow, without needing to edit any classical functions or compile to Wasm.

## Setup

To create and use your own Hybrid Compute functions in your workflow, it is necessary to use Wasm and a language that compiles code to Wasm. These instructions contain the steps for setting up an environment to do this.

It is recommended to either use Rust or C++ for the compilation language to Wasm. The languages have functionality that easily compiles to Wasm. For other language options see [Wasm Getting Started Developer's page](https://webassembly.org/getting-started/developers-guide/).

### Option 1: Rust

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

#### Create classical functions in Rust

Create the classical functions you plan to use in your hybrid compute program in the `src/lib.rs` file. Examples of the types of functions that can be put in Rust code are provided in the `Rust` folders within each of the example folders. *We recommend looking at one of the provided examples to know which functions and attributes are required for Hybrid Compute to work.*

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

In order to use the compiled wasm from your quantum program, you will want to find the associated `.wasm` file that the compilation step created. This can be found in the `./target/wasm32-unknown-unknown/release` folder of your project. You'll note a file with the name of your project with the extension `.wasm`. To use this in your quantum program, you can copy-paste it into another folder, or ensure that the file path for your Wams file handler points to this location.

## Wasm Usage

Some limitations exist on what kinds of functions or capabilities are enabled. 

1. Users can submit Wasm files generated from Rust or other languages so long as they don’t use Wasi features or break the Wasm memory sandbox. Random number generation* and file I/O are not allowed.
2. Quantum programs can only call Wasm functions that accept multiple integers and return an integer. 
    * For example, examine the call `x = wasm_func(a, b, c);` The `wasm_func` function can exist in QASM where `a`, `b`, `c`, and `x` are classical registers. 
    * Such function calls do have timing restrictions. Wasm calls in quantum programs have been run that took multiple milliseconds.
3. The state of a Wasm program persists between calls in the quantum programs within the limits of the chunking window (~300 shots). 
    * This means global variables can hold mutable state that is modified between calls.
    * One can imagine using various global arrays. It's possible to have global pointers work with heap allocated data structures (for example using unsafe Rust features when compiling to Wasm).
4. Quantum + Web Assembly programs have to come in under 6 MB*, currently.


---
