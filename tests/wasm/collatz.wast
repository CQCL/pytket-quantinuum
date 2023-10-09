(module
  (type (;0;) (func))
  (type (;1;) (func (param i32) (result i32)))
  (func $__wasm_call_ctors (type 0))
  (func $init (type 0)
    return)
  (func $collatz (type 1) (param i32) (result i32)
    (local i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
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
    i32.store offset=8
    local.get 3
    i32.load offset=8
    local.set 4
    block  ;; label = @1
      block  ;; label = @2
        local.get 4
        br_if 0 (;@2;)
        i32.const 0
        local.set 5
        local.get 3
        local.get 5
        i32.store offset=12
        br 1 (;@1;)
      end
      i32.const 0
      local.set 6
      local.get 3
      local.get 6
      i32.store offset=4
      block  ;; label = @2
        loop  ;; label = @3
          local.get 3
          i32.load offset=8
          local.set 7
          i32.const 1
          local.set 8
          local.get 7
          local.set 9
          local.get 8
          local.set 10
          local.get 9
          local.get 10
          i32.ne
          local.set 11
          i32.const 1
          local.set 12
          local.get 11
          local.get 12
          i32.and
          local.set 13
          local.get 13
          i32.eqz
          br_if 1 (;@2;)
          local.get 3
          i32.load offset=8
          local.set 14
          i32.const 2
          local.set 15
          local.get 14
          local.get 15
          i32.and
          local.set 16
          block  ;; label = @4
            block  ;; label = @5
              local.get 16
              i32.eqz
              br_if 0 (;@5;)
              local.get 3
              i32.load offset=8
              local.set 17
              i32.const 3
              local.set 18
              local.get 17
              local.get 18
              i32.mul
              local.set 19
              i32.const 1
              local.set 20
              local.get 19
              local.get 20
              i32.add
              local.set 21
              i32.const 1
              local.set 22
              local.get 21
              local.get 22
              i32.shr_u
              local.set 23
              local.get 3
              local.get 23
              i32.store offset=8
              br 1 (;@4;)
            end
            local.get 3
            i32.load offset=8
            local.set 24
            i32.const 1
            local.set 25
            local.get 24
            local.get 25
            i32.shr_u
            local.set 26
            local.get 3
            local.get 26
            i32.store offset=8
          end
          local.get 3
          i32.load offset=4
          local.set 27
          i32.const 1
          local.set 28
          local.get 27
          local.get 28
          i32.add
          local.set 29
          local.get 3
          local.get 29
          i32.store offset=4
          br 0 (;@3;)
        end
      end
      local.get 3
      i32.load offset=4
      local.set 30
      local.get 3
      local.get 30
      i32.store offset=12
    end
    local.get 3
    i32.load offset=12
    local.set 31
    local.get 31
    return)
  (memory (;0;) 2)
  (global $__stack_pointer (mut i32) (i32.const 66560))
  (global (;1;) i32 (i32.const 1024))
  (global (;2;) i32 (i32.const 1024))
  (global (;3;) i32 (i32.const 1024))
  (global (;4;) i32 (i32.const 66560))
  (global (;5;) i32 (i32.const 0))
  (global (;6;) i32 (i32.const 1))
  (export "memory" (memory 0))
  (export "__wasm_call_ctors" (func $__wasm_call_ctors))
  (export "init" (func $init))
  (export "collatz" (func $collatz))
  (export "__dso_handle" (global 1))
  (export "__data_end" (global 2))
  (export "__global_base" (global 3))
  (export "__heap_base" (global 4))
  (export "__memory_base" (global 5))
  (export "__table_base" (global 6)))
