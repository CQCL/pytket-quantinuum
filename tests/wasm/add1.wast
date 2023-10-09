(module
  (type (;0;) (func))
  (type (;1;) (func (param i32) (result i32)))
  (type (;2;) (func (param i32 i32) (result i32)))
  (type (;3;) (func (param i64) (result i64)))
  (type (;4;) (func (param i32)))
  (type (;5;) (func (result i32)))
  (func $init (type 0))
  (func $add_one (type 1) (param i32) (result i32)
    local.get 0
    i32.const 1
    i32.add)
  (func $multi (type 2) (param i32 i32) (result i32)
    local.get 1
    local.get 0
    i32.mul)
  (func $add_two (type 1) (param i32) (result i32)
    local.get 0
    i32.const 2
    i32.add)
  (func $add_something (type 3) (param i64) (result i64)
    local.get 0
    i64.const 11
    i64.add)
  (func $add_eleven (type 1) (param i32) (result i32)
    local.get 0
    i32.const 11
    i32.add)
  (func $no_return (type 4) (param i32))
  (func $no_parameters (type 5) (result i32)
    i32.const 11)
  (func $new_function (type 5) (result i32)
    i32.const 13)
  (table (;0;) 1 1 funcref)
  (memory (;0;) 16)
  (global $__stack_pointer (mut i32) (i32.const 1048576))
  (global (;1;) i32 (i32.const 1048576))
  (global (;2;) i32 (i32.const 1048576))
  (export "memory" (memory 0))
  (export "init" (func $init))
  (export "add_one" (func $add_one))
  (export "multi" (func $multi))
  (export "add_two" (func $add_two))
  (export "add_something" (func $add_something))
  (export "add_eleven" (func $add_eleven))
  (export "no_return" (func $no_return))
  (export "no_parameters" (func $no_parameters))
  (export "new_function" (func $new_function))
  (export "__data_end" (global 1))
  (export "__heap_base" (global 2)))
