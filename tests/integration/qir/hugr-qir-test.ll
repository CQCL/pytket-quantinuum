; ModuleID = 'hugr-qir'
source_filename = "hugr-qir"

%QUBIT = type opaque
%RESULT = type opaque

@0 = private unnamed_addr constant [2 x i8] c"0\00", align 1

define void @__hugr__.main.1() #0 {
alloca_block:
  call void @__quantum__qis__phasedx__body(double 0x3FF921FB54442D18, double 0xBFF921FB54442D18, %QUBIT* null)
  call void @__quantum__qis__rz__body(double 0x400921FB54442D18, %QUBIT* null)
  %0 = call %RESULT* @__quantum__qis__mz__body(%QUBIT* null)
  %1 = call i1 @__quantum__qis__read_result__body(%RESULT* %0)
  call void @__quantum__qis__phasedx__body(double 0x3FF921FB54442D18, double 0xBFF921FB54442D18, %QUBIT* inttoptr (i64 1 to %QUBIT*))
  call void @__quantum__qis__rz__body(double 0x400921FB54442D18, %QUBIT* inttoptr (i64 1 to %QUBIT*))
  %2 = call %RESULT* @__quantum__qis__mz__body(%QUBIT* inttoptr (i64 1 to %QUBIT*))
  %3 = call i1 @__quantum__qis__read_result__body(%RESULT* %2)
  %4 = xor i1 %3, %1
  call void @__quantum__rt__bool_record_output(i1 %4, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @0, i32 0, i32 0))
  ret void
}

declare void @__quantum__qis__phasedx__body(double, double, %QUBIT*)

declare void @__quantum__qis__rz__body(double, %QUBIT*)

declare %RESULT* @__quantum__qis__mz__body(%QUBIT*)

declare i1 @__quantum__qis__read_result__body(%RESULT*)

declare void @__quantum__rt__bool_record_output(i1, i8*)

attributes #0 = { "entry_point" "output_labeling_schema" "qir_profiles"="custom" "required_num_qubits"="2" "required_num_results"="2" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"qir_major_version", i32 1}
!1 = !{i32 7, !"qir_minor_version", i32 0}
!2 = !{i32 1, !"dynamic_qubit_management", i1 false}
!3 = !{i32 1, !"dynamic_result_management", i1 false}
