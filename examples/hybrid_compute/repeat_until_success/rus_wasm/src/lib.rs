// Global variable (mutable) since it will track totals over time
static mut TOTAL_MEAS:i32 = 0;

// Define array for tracking measurements
const ARRAY_SIZE: usize = 20;
static mut MEAS_ARRAY: [i32; ARRAY_SIZE] = [0; ARRAY_SIZE];

// For hybrid compute to work, [no_mangle] must be at the top of all functions
// we plan to call from our quantum program
// For more info see: [Calling Rust from Other Languages]
// (https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html?highlight=no_mangle#calling-rust-functions-from-other-languages)
#[no_mangle]
fn init(){
    // This function can have nothing it in, or load some initial function. 
    // It is needed when passed via the Quantinuum API to warm up the wasm execution environment. 
    // It can also be used to set up a global state.
}

#[no_mangle]
fn add_count(meas:i32, count:usize) -> i32 {
    // The unsafe keyword is needed to modify global variables.
    // For more information on the unsafe Rust feature, see: https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html
    unsafe{
        // Track the total of the measurements, for the RUS criteria
        TOTAL_MEAS += meas;

        // Track the measurements
        MEAS_ARRAY[count] = meas;

        // Return the condition for success back our quantum program
        let cond = TOTAL_MEAS;
        return cond 
    }
}

// The [test] attribute indicate which functions are tests
#[test]
fn Test_Rust_code(){
    add_count(0,0);
    add_count(1,1);
    add_count(0,2);
    unsafe{
        println!("The total is {:?}", TOTAL_MEAS);
        println!("The qubit measurement array is {:?}", MEAS_ARRAY);
    }
}
