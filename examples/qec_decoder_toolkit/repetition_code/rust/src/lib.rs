// Copyright 2020-2024 Quantinuum
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

static mut SYN_OLD: i32 = 0;

#[no_mangle]
fn init(){
    // This function can have nothing it in, or load some initial function. 
    // It is needed when passed via the Quantinuum API to warm up the wasm execution environment. 
    // It can also be used to set up a global state.
}

#[no_mangle]
fn decode3(syn: i32) -> i32  { //takes in a string and returns and a string
    let mut decoder:HashMap<i32,i32> = HashMap::new();
    decoder.insert(1, 1); //001 = 1, if syn = 1 then error on qubit 0
    decoder.insert(3, 2); //010 = 2, if sny = 3 then error on qubit 1
    decoder.insert(2, 4); //100 = 4, if syn = 2 then error on qubit 2

    let pfu = 0; //Define a register to hold our correction

    unsafe{
        let syn_new: i32 = SYN_OLD ^ syn;
        SYN_OLD = syn;
        println!("{}", syn_new);

        if syn == 0 {
            return pfu;
        }
        else {
            return pfu ^ decoder[&syn_new];
        }
    }
}

#[no_mangle]
fn reset(){
  unsafe{
    SYN_OLD = 0;
  }
  
}

// #[test]
// fn test_decode3() {
//     let val = decode3(3, 1);
//     unsafe{
//         println!("{}", val);
//     }
// }