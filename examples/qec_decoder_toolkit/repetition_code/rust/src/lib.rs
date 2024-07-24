use std::collections::HashMap;

static mut SYN_OLD: i32 = 0;

#[no_mangle]
fn init(){
    // This function can have nothing it in, or load some initial function. 
    // It is needed when passed via the Quantinuum API to warm up the wasm execution environment. 
    // It can also be used to set up a global state.
}

#[no_mangle]
fn decode3(syn: i32, pfu: i32) -> i32  { //takes in a string and returns and a string
    let mut decoder:HashMap<i32,i32> = HashMap::new();

    // let syn_new: i32 = SYN_OLD ^ syn;
    decoder.insert(1, 1); //001 = 1, if syn = 1 then error on qubit 0
    decoder.insert(3, 2); //010 = 2, if sny = 3 then error on qubit 1
    decoder.insert(2, 4); //100 = 4, if syn = 2 then error on qubit 2

    unsafe {
        SYN_OLD = syn;
        println!("SYN_OLD: {}, syn: {}", SYN_OLD, syn)
    }

    if syn == 0 {
        return pfu;
    }
    else {
        return pfu ^ decoder[&syn];
    }
}

fn main() -> i32 {
    let pfu: i32 = decode3(1, 2);
    return 0;
}

// #[test]
// fn test_decode3() {
//     let val = decode3(3, 1);
//     unsafe{
//         println!("{}", val);
//     }
// }