// RUN: circt-opt --pass-pipeline='builtin.module(hw.module(hw-uhdi-verilog-snapshot))' %s | FileCheck %s

// The pass only acts on dbg.* ops that already carry uhdi.stable_id
// (a stamp placed by firrtl-uhdi-init earlier in the pipeline). For each,
// it records the Verilog-side signal name as
// `uhdi.repr_entry = {verilog = {name = "..."}}`.

// CHECK-LABEL: hw.module @Ports
hw.module @Ports(in %a : i8, in %b : i4, out o : i12) {
  // Block argument -> port name; repr_entry gets stamped.
  // CHECK: dbg.variable "a", %a {uhdi.repr_entry = {verilog = {name = "a"}}, uhdi.stable_id = "var_aaaa_0000"}
  dbg.variable "a", %a {uhdi.stable_id = "var_aaaa_0000"} : i8

  // hw.wire with hw.verilogName -> that name.
  // CHECK: dbg.variable "myWire", %w_inner {uhdi.repr_entry = {verilog = {name = "w_rename"}}, uhdi.stable_id = "var_bbbb_0000"}
  %w_inner = hw.wire %a {hw.verilogName = "w_rename"} : i8
  dbg.variable "myWire", %w_inner {uhdi.stable_id = "var_bbbb_0000"} : i8

  // No stable_id -> pass is a no-op.
  // CHECK: dbg.variable "untagged", %b : i4
  dbg.variable "untagged", %b : i4

  // Intermediate signal (comb.concat result) whose value reaches
  // hw.output resolves to the destination port name -- matches the
  // output-port fallback in findVerilogName.
  // CHECK: dbg.variable "intermediate", %0 {uhdi.repr_entry = {verilog = {name = "o"}}, uhdi.stable_id = "var_cccc_0000"}
  %0 = comb.concat %a, %b : i8, i4
  dbg.variable "intermediate", %0 {uhdi.stable_id = "var_cccc_0000"} : i12

  hw.output %0 : i12
}

// CHECK-LABEL: hw.module @ChildScope
hw.module private @Child(in %x : i8, out y : i8) {
  hw.output %x : i8
}

hw.module @ChildScope(in %a : i8, out b : i8) {
  // dbg.scope resolves via the matching hw.instance's hw.verilogName.
  // CHECK: dbg.scope "c", "Child" {uhdi.repr_entry = {verilog = {name = "c_inst"}}, uhdi.stable_id = "scope_dddd_0000"}
  %s = dbg.scope "c", "Child" {uhdi.stable_id = "scope_dddd_0000"}
  %c.y = hw.instance "c" @Child(x: %a: i8) -> (y: i8) {hw.verilogName = "c_inst"}
  hw.output %c.y : i8
}

// CHECK-LABEL: hw.module @Idempotent
hw.module @Idempotent(in %p : i1) {
  // Pre-existing repr_entry is not rewritten.
  // CHECK: dbg.variable "p", %p {uhdi.repr_entry = {verilog = {name = "preset"}}, uhdi.stable_id = "var_eeee_0000"}
  dbg.variable "p", %p {
    uhdi.stable_id = "var_eeee_0000",
    uhdi.repr_entry = {verilog = {name = "preset"}}
  } : i1
  hw.output
}
