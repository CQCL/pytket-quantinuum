; ModuleID = 'test_pytket_qir_wasm_5-QIRProfile.PYTKET'
source_filename = "test_pytket_qir_wasm_5-QIRProfile.PYTKET"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [2 x i8] c"c\00"
@1 = internal constant [3 x i8] c"c0\00"
@2 = internal constant [3 x i8] c"c1\00"

define void @main() #0 {
entry:
  %0 = call i1* @create_creg(i64 6)
  %1 = call i1* @create_creg(i64 64)
  %2 = call i1* @create_creg(i64 32)
  call void @mz_to_creg_bit(%Qubit* null, i1* %0, i64 0)
  call void @mz_to_creg_bit(%Qubit* inttoptr (i64 1 to %Qubit*), i1* %0, i64 1)
  call void @mz_to_creg_bit(%Qubit* inttoptr (i64 2 to %Qubit*), i1* %0, i64 2)
  call void @set_creg_bit(i1* %1, i64 0, i1 true)
  call void @set_creg_bit(i1* %1, i64 1, i1 false)
  call void @set_creg_bit(i1* %1, i64 2, i1 false)
  call void @set_creg_bit(i1* %1, i64 3, i1 false)
  call void @set_creg_bit(i1* %1, i64 4, i1 false)
  call void @set_creg_bit(i1* %1, i64 5, i1 false)
  call void @set_creg_bit(i1* %1, i64 6, i1 false)
  call void @set_creg_bit(i1* %1, i64 7, i1 false)
  call void @set_creg_bit(i1* %1, i64 8, i1 false)
  call void @set_creg_bit(i1* %1, i64 9, i1 false)
  call void @set_creg_bit(i1* %1, i64 10, i1 false)
  call void @set_creg_bit(i1* %1, i64 11, i1 false)
  call void @set_creg_bit(i1* %1, i64 12, i1 false)
  call void @set_creg_bit(i1* %1, i64 13, i1 false)
  call void @set_creg_bit(i1* %1, i64 14, i1 false)
  call void @set_creg_bit(i1* %1, i64 15, i1 false)
  call void @set_creg_bit(i1* %1, i64 16, i1 false)
  call void @set_creg_bit(i1* %1, i64 17, i1 false)
  call void @set_creg_bit(i1* %1, i64 18, i1 false)
  call void @set_creg_bit(i1* %1, i64 19, i1 false)
  call void @set_creg_bit(i1* %1, i64 20, i1 false)
  call void @set_creg_bit(i1* %1, i64 21, i1 false)
  call void @set_creg_bit(i1* %1, i64 22, i1 false)
  call void @set_creg_bit(i1* %1, i64 23, i1 false)
  call void @set_creg_bit(i1* %1, i64 24, i1 false)
  call void @set_creg_bit(i1* %1, i64 25, i1 false)
  call void @set_creg_bit(i1* %1, i64 26, i1 false)
  call void @set_creg_bit(i1* %1, i64 27, i1 false)
  call void @set_creg_bit(i1* %1, i64 28, i1 false)
  call void @set_creg_bit(i1* %1, i64 29, i1 false)
  call void @set_creg_bit(i1* %1, i64 30, i1 false)
  call void @set_creg_bit(i1* %1, i64 31, i1 false)
  call void @set_creg_bit(i1* %1, i64 32, i1 false)
  call void @set_creg_bit(i1* %1, i64 33, i1 false)
  call void @set_creg_bit(i1* %1, i64 34, i1 false)
  call void @set_creg_bit(i1* %1, i64 35, i1 false)
  call void @set_creg_bit(i1* %1, i64 36, i1 false)
  call void @set_creg_bit(i1* %1, i64 37, i1 false)
  call void @set_creg_bit(i1* %1, i64 38, i1 false)
  call void @set_creg_bit(i1* %1, i64 39, i1 false)
  call void @set_creg_bit(i1* %1, i64 40, i1 false)
  call void @set_creg_bit(i1* %1, i64 41, i1 false)
  call void @set_creg_bit(i1* %1, i64 42, i1 false)
  call void @set_creg_bit(i1* %1, i64 43, i1 false)
  call void @set_creg_bit(i1* %1, i64 44, i1 false)
  call void @set_creg_bit(i1* %1, i64 45, i1 false)
  call void @set_creg_bit(i1* %1, i64 46, i1 false)
  call void @set_creg_bit(i1* %1, i64 47, i1 false)
  call void @set_creg_bit(i1* %1, i64 48, i1 false)
  call void @set_creg_bit(i1* %1, i64 49, i1 false)
  call void @set_creg_bit(i1* %1, i64 50, i1 false)
  call void @set_creg_bit(i1* %1, i64 51, i1 false)
  call void @set_creg_bit(i1* %1, i64 52, i1 false)
  call void @set_creg_bit(i1* %1, i64 53, i1 false)
  call void @set_creg_bit(i1* %1, i64 54, i1 false)
  call void @set_creg_bit(i1* %1, i64 55, i1 false)
  call void @set_creg_bit(i1* %1, i64 56, i1 false)
  call void @set_creg_bit(i1* %1, i64 57, i1 false)
  call void @set_creg_bit(i1* %1, i64 58, i1 false)
  call void @set_creg_bit(i1* %1, i64 59, i1 false)
  call void @set_creg_bit(i1* %1, i64 60, i1 false)
  call void @set_creg_bit(i1* %1, i64 61, i1 false)
  call void @set_creg_bit(i1* %1, i64 62, i1 false)
  call void @set_creg_bit(i1* %1, i64 63, i1 false)
  %3 = call i64 @get_int_from_creg(i1* %1)
  %4 = call i64 @add_one(i64 %3)
  call void @set_creg_to_int(i1* %2, i64 %4)
  %5 = call i64 @get_int_from_creg(i1* %0)
  call void @__quantum__rt__int_record_output(i64 %5, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @0, i32 0, i32 0))
  %6 = call i64 @get_int_from_creg(i1* %1)
  call void @__quantum__rt__int_record_output(i64 %6, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @1, i32 0, i32 0))
  %7 = call i64 @get_int_from_creg(i1* %2)
  call void @__quantum__rt__int_record_output(i64 %7, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @2, i32 0, i32 0))
  ret void
}

declare i1 @__quantum__qis__read_result__body(%Result*)

declare void @__quantum__rt__int_record_output(i64, i8*)

declare void @init() #1

declare i64 @add_one(i64) #1

declare i64 @multi(i64, i64) #1

declare i64 @add_two(i64) #1

declare i64 @add_eleven(i64) #1

declare void @no_return(i64) #1

declare i64 @no_parameters() #1

declare i64 @new_function() #1

declare i1 @get_creg_bit(i1*, i64)

declare void @set_creg_bit(i1*, i64, i1)

declare void @set_creg_to_int(i1*, i64)

declare i1* @create_creg(i64)

declare i64 @get_int_from_creg(i1*)

declare void @mz_to_creg_bit(%Qubit*, i1*, i64)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="6" "required_num_results"="6" }

attributes #1 = { "wasm" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
