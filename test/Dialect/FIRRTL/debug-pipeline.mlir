// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-materialize-debug-info,firrtl-lower-intrinsics)))' %s | FileCheck %s

// Integration check: when MaterializeDebugInfo runs BEFORE LowerIntrinsics
// (the order firtool pins in lib/Firtool/Firtool.cpp), names covered by a
// `circt_debug_var` intrinsic must end up with a single rich `dbg.variable`
// from LowerIntrinsics -- not a duplicate basic one from MaterializeDebugInfo.

// CHECK-LABEL: firrtl.module @PipelineDedup
firrtl.circuit "PipelineDedup" {
  firrtl.module @PipelineDedup() {
    %w = firrtl.wire : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "w", typeName: none = "UInt">
      %w : (!firrtl.uint<8>) -> ()

    %u = firrtl.wire : !firrtl.uint<4>

    // The rich variable for `w` (with typeName) is the only `dbg.variable "w"`.
    // CHECK:     dbg.variable "w", %{{.*}} {typeName = "UInt"} : !firrtl.uint<8>
    // CHECK-NOT: dbg.variable "w"
    //
    // The wire `u` has no intrinsic, so MaterializeDebugInfo emits a basic
    // `dbg.variable "u"` (no typeName, ground type passes through).
    // CHECK:     dbg.variable "u", %{{.*}} : !firrtl.uint<4>
    // CHECK-NOT: dbg.variable "u"
  }
}
