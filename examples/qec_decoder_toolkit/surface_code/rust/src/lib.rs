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


static mut PFU:i32 = 0; // global variable to track the value of the Pauli Frame Update (000000000)


#[no_mangle]
fn init(){
    // This function can have nothing it in, or load some initial function. 
    // It is needed when passed via the Quantinuum API to warm up the wasm execution environment. 
    // It can also be used to set up a global state.
}


#[no_mangle]
fn set_pfu_value(syn: i32) {
    unsafe{
        if syn == 1 {
            PFU = 4; // 000000100
        }
        if syn == 2 {
            PFU = 128; // 010000000
        }
        if syn == 4 {
            PFU = 1; // 000000001
        }
        if syn == 8 {
            PFU = 64; // 001000000
        }
        if syn == 3 {
            PFU = 32; // 000100000
        }
        if syn == 6 {
            PFU = 16; // 000010000
        }
        if syn == 12 {
            PFU = 8; // 000001000
        }
        if syn == 5 {
            PFU = 48; // 000110000
        }
        if syn == 10 {
            PFU = 24; // 000011000
        }
        if syn == 9 {
            PFU = 68; // 001000100
        }
        if syn == 7 {
            PFU = 33; // 000100001
        }
        if syn == 11 {
            PFU = 96; // 001100000
        }
        if syn == 13 {
            PFU = 12; // 000001100
        }
        if syn == 14 {
            PFU = 136; // 010001000
        }
        if syn == 15 {
            PFU = 40; // 000101000
        }
    }
}


#[no_mangle]
fn update_pfu(syn: i32) {
    unsafe{
        let pfu_temp: i32 = PFU.clone();
        set_pfu_value(syn);
        PFU = pfu_temp ^ PFU;
    }
}


#[no_mangle]
fn get_pfu() -> i32 {
    unsafe{
        return PFU;
    }
}


#[no_mangle]
fn reset_pfu() {
    unsafe{
        PFU = 0; // 000000000
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        reset_pfu();
        let val: i32 = 15;
        set_pfu_value(val);
        assert_eq!(get_pfu(), 40);
    }

    #[test]
    fn test2() {
        reset_pfu();
        let val: i32 = 4;
        set_pfu_value(val);
        assert_eq!(get_pfu(), 1);
    }

    #[test]
    fn test3() {
        set_pfu_value(5);
        reset_pfu();
        assert_eq!(get_pfu(), 0);
    }
}