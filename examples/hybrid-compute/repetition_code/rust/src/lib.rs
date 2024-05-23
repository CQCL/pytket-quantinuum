use std::collections::HashMap;
use std::fmt;

#[no_mangle]
fn decode3(syn: i32, mut PFU: i32) -> i32  { //takes in a string and returns and a string
    let mut decoder:HashMap<i32,i32> = HashMap::new();
    let mut PFU_new = 0;

    decoder.insert(1, 1); //001 = 1, if syn = 1 then error on qubit 0
    decoder.insert(3, 2); //010 = 2, if sny = 3 then error on qubit 1
    decoder.insert(2, 4); //100 = 4, if syn = 2 then error on qubit 2

    PFU_new = PFU^decoder[&syn];

    return PFU_new
}
