(module
  (type (;0;) (func))
  (type (;1;) (func (param i32)))
  (type (;2;) (func (result i32)))
  (func $__wasm_call_ctors (type 0))
  (func $init (type 0)
    return)
  (func $set_c (type 1) (param i32)
    (local i32 i32 i32 i32 i32)
    global.get $__stack_pointer
    local.set 1
    i32.const 16
    local.set 2
    local.get 1
    local.get 2
    i32.sub
    local.set 3
    local.get 3
    local.get 0
    i32.store offset=12
    local.get 3
    i32.load offset=12
    local.set 4
    i32.const 0
    local.set 5
    local.get 5
    local.get 4
    i32.store offset=1024
    return)
  (func $conditional_increment_c (type 1) (param i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
    global.get $__stack_pointer
    local.set 1
    i32.const 16
    local.set 2
    local.get 1
    local.get 2
    i32.sub
    local.set 3
    local.get 3
    local.get 0
    i32.store offset=12
    local.get 3
    i32.load offset=12
    local.set 4
    block  ;; label = @1
      local.get 4
      i32.eqz
      br_if 0 (;@1;)
      i32.const 0
      local.set 5
      local.get 5
      i32.load offset=1024
      local.set 6
      i32.const 1
      local.set 7
      local.get 6
      local.get 7
      i32.add
      local.set 8
      i32.const 0
      local.set 9
      local.get 9
      local.get 8
      i32.store offset=1024
    end
    return)
  (func $get_c (type 2) (result i32)
    (local i32 i32)
    i32.const 0
    local.set 0
    local.get 0
    i32.load offset=1024
    local.set 1
    local.get 1
    return)
  (memory (;0;) 2)
  (global $__stack_pointer (mut i32) (i32.const 66576))
  (global (;1;) i32 (i32.const 1024))
  (global (;2;) i32 (i32.const 1028))
  (global (;3;) i32 (i32.const 1024))
  (global (;4;) i32 (i32.const 66576))
  (global (;5;) i32 (i32.const 0))
  (global (;6;) i32 (i32.const 1))
  (export "memory" (memory 0))
  (export "__wasm_call_ctors" (func $__wasm_call_ctors))
  (export "init" (func $init))
  (export "set_c" (func $set_c))
  (export "conditional_increment_c" (func $conditional_increment_c))
  (export "get_c" (func $get_c))
  (export "__dso_handle" (global 1))
  (export "__data_end" (global 2))
  (export "__global_base" (global 3))
  (export "__heap_base" (global 4))
  (export "__memory_base" (global 5))
  (export "__table_base" (global 6)))
