; ModuleID = 'test_pytket_qir_wasm_5-QIRProfile.ADAPTIVE'
source_filename = "test_pytket_qir_wasm_5-QIRProfile.ADAPTIVE"

%Qubit = type opaque
%Result = type opaque

@0 = internal constant [2 x i8] c"c\00"
@1 = internal constant [3 x i8] c"c0\00"
@2 = internal constant [3 x i8] c"c1\00"

define void @main() #0 {
entry:
  call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)
  %0 = call i1 @__quantum__qis__read_result__body(%Result* null)
  %1 = zext i1 %0 to i64
  %2 = mul i64 %1, 1
  %3 = or i64 %2, 0
  %4 = sub i64 1, %1
  %5 = mul i64 %4, 1
  %6 = xor i64 9223372036854775807, %5
  %7 = and i64 %6, %3
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* inttoptr (i64 1 to %Result*))
  %8 = call i1 @__quantum__qis__read_result__body(%Result* inttoptr (i64 1 to %Result*))
  %9 = zext i1 %8 to i64
  %10 = mul i64 %9, 2
  %11 = or i64 %10, %7
  %12 = sub i64 1, %9
  %13 = mul i64 %12, 2
  %14 = xor i64 9223372036854775807, %13
  %15 = and i64 %14, %11
  call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 2 to %Qubit*), %Result* inttoptr (i64 2 to %Result*))
  %16 = call i1 @__quantum__qis__read_result__body(%Result* inttoptr (i64 2 to %Result*))
  %17 = zext i1 %16 to i64
  %18 = mul i64 %17, 4
  %19 = or i64 %18, %15
  %20 = sub i64 1, %17
  %21 = mul i64 %20, 4
  %22 = xor i64 9223372036854775807, %21
  %23 = and i64 %22, %19
  %24 = call i64 @add_one(i64 1)
  call void @__quantum__rt__int_record_output(i64 %23, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @0, i32 0, i32 0))
  call void @__quantum__rt__int_record_output(i64 1, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @1, i32 0, i32 0))
  call void @__quantum__rt__int_record_output(i64 %24, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @2, i32 0, i32 0))
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

declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) #2

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="6" "required_num_results"="6" }
attributes #1 = { "wasm" }
attributes #2 = { "irreversible" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
