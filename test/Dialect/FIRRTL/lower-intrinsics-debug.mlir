// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s --split-input-file | FileCheck %s

// Scalar, bundle, and vec cases for the three Chisel debug intrinsics
// (circt_debug_moduleinfo, circt_debug_var, circt_debug_subfield) and
// circt_debug_enumdef.

// CHECK-LABEL: firrtl.module @DebugTest
firrtl.circuit "DebugTest" {
  firrtl.module @DebugTest(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {

    // 1. moduleinfo (empty params array is forwarded as [])
    // CHECK-DAG: dbg.moduleinfo {params = [], typeName = "MyModule"}
    firrtl.int.generic "circt_debug_moduleinfo"
      <typeName: none = "MyModule", params: none = "[]"> : () -> ()

    // 2. enumdef with JSON array (value serialized as string by frontend)
    // CHECK-DAG: dbg.enumdef "MyState", fqn "pkg.MyState$", {Idle = 0 : i64, Run = 1 : i64}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":\"0\"},{\"name\":\"Run\",\"value\":\"1\"}]">
      : () -> ()

    // 3. circt_debug_var on scalar wire -> dbg.variable with the wire value
    // CHECK: dbg.variable "w", %{{.*}} {typeName = "UInt"} : !firrtl.uint<8>
    %w = firrtl.wire : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "w", typeName: none = "UInt">
      %w : (!firrtl.uint<8>) -> ()

    // 4. circt_debug_var on a passive bundle -> dbg.struct + dbg.variable
    //    The converter builds:
    //      %a = firrtl.subfield %io[in]
    //      %b = firrtl.subfield %io[out]
    //      %s = dbg.struct {"in": %a, "out": %b}
    //      dbg.variable "io", %s
    // CHECK:      firrtl.subfield %io[in]
    // CHECK:      firrtl.subfield %io[out]
    // CHECK:      dbg.struct
    // CHECK: dbg.variable "io", %{{.*}} {typeName = "MyBundle"}
    %io = firrtl.wire : !firrtl.bundle<in: uint<8>, out: uint<8>>
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<in: uint<8>, out: uint<8>>) -> ()

    // 5. circt_debug_var on a Vec -> dbg.array + dbg.variable
    // CHECK:      firrtl.subindex %v[0]
    // CHECK:      firrtl.subindex %v[1]
    // CHECK:      dbg.array
    // CHECK-NEXT: dbg.variable "v", %{{.*}} {typeName = "Vec"}
    %v = firrtl.wire : !firrtl.vector<uint<4>, 2>
    firrtl.int.generic "circt_debug_var"
      <name: none = "v", typeName: none = "Vec">
      %v : (!firrtl.vector<uint<4>, 2>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Bundle with an enum-typed field -- the field must get a dbg.subfield
// wrapping it with an enumDef reference.
//
// Chisel emits:
//   circt_debug_enumdef for MyState
//   circt_debug_subfield(parent="io", name="state") for io.state with enumFqn
//   circt_debug_subfield(parent="io", name="data")  for io.data
//   circt_debug_var(name="io")                      for io

// CHECK-LABEL: firrtl.module @BundleEnumFieldTest
firrtl.circuit "BundleEnumFieldTest" {
  firrtl.module @BundleEnumFieldTest(
      in %in: !firrtl.uint<8>,
      out %out: !firrtl.uint<8>) {

    // enumdef must appear in the output
    // CHECK: %[[E:.+]] = dbg.enumdef "MyState", fqn "pkg.MyState$", {Idle = 0 : i64, Run = 1 : i64}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":\"0\"},{\"name\":\"Run\",\"value\":\"1\"}]">
      : () -> ()

    %io = firrtl.wire : !firrtl.bundle<state: uint<2>, data: uint<8>>

    // Leaf io.state with enumFqn -- captured into leafMetaMap, erased
    // CHECK-NOT: circt_debug_subfield
    // CHECK-NOT: dbg.variable "state"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.state", typeName: none = "UInt", parent: none = "io",
       enumTypeName: none = "MyState", enumFqn: none = "pkg.MyState$">
      %io : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    // Leaf io.data without enum
    // CHECK-NOT: dbg.variable "data"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.data", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    // Root circt_debug_var -- produces dbg.struct + dbg.variable
    // CHECK:      firrtl.subfield %{{.*}}[state]
    // CHECK:      dbg.subfield "io.state", %{{.*}} enumDef %[[E]] {typeName = "UInt"} : !firrtl.uint<2>
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK:      dbg.subfield "io.data", %{{.*}} {typeName = "UInt"} : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io", %{{.*}} {typeName = "MyBundle"}
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Bundle with a FixedPoint field carrying type parameters.
// Verifies that the "params" JSON string is parsed and forwarded to
// dbg.subfield as an ArrayAttr of DictionaryAttrs.

// CHECK-LABEL: firrtl.module @BundleParamsFieldTest
firrtl.circuit "BundleParamsFieldTest" {
  firrtl.module @BundleParamsFieldTest(
      in %in: !firrtl.uint<8>,
      out %out: !firrtl.uint<8>) {

    %io = firrtl.wire : !firrtl.bundle<fp: uint<8>, data: uint<8>>

    // Leaf io.fp with params
    // CHECK-NOT: dbg.variable "fp"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.fp", typeName: none = "FixedPoint", parent: none = "io",
       params: none = "[{\"name\":\"width\",\"value\":\"8\"},{\"name\":\"binaryPoint\",\"value\":\"4\"}]">
      %io : (!firrtl.bundle<fp: uint<8>, data: uint<8>>) -> ()

    // Leaf io.data without params
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.data", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<fp: uint<8>, data: uint<8>>) -> ()

    // Root circt_debug_var -- produces dbg.struct + dbg.variable
    // CHECK:      firrtl.subfield %{{.*}}[fp]
    // CHECK:      dbg.subfield "io.fp", %{{.*}} {params = [{name = "width", value = "8"}, {name = "binaryPoint", value = "4"}], typeName = "FixedPoint"} : !firrtl.uint<8>
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK:      dbg.subfield "io.data", %{{.*}} {typeName = "UInt"} : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io", %{{.*}} {typeName = "MyBundle"}
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<fp: uint<8>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// moduleinfo with constructor params

// CHECK-LABEL: firrtl.module @ModuleInfoWithParams
firrtl.circuit "ModuleInfoWithParams" {
  firrtl.module @ModuleInfoWithParams() {
    // CHECK: dbg.moduleinfo {params = [{name = "width", value = "8"}], typeName = "Counter"}
    firrtl.int.generic "circt_debug_moduleinfo"
      <typeName: none = "Counter",
       params: none = "[{\"name\":\"width\",\"value\":\"8\"}]">
      : () -> ()
  }
}

// -----

// 0-operand circt_debug_var: the frontend emits no SSA operand when the
// referenced value is a non-passive aggregate (e.g. a bidirectional bundle
// port) whose SSA form is not directly assignable to a dbg.variable. The
// converter must locate the matching port / wire / node / reg by name.

// CHECK-LABEL: firrtl.module @ZeroOperandVarPort
firrtl.circuit "ZeroOperandVarPort" {
  firrtl.module @ZeroOperandVarPort(
      in %a: !firrtl.uint<8>,
      out %b: !firrtl.uint<8>) {
    // Match by port name; resolves to the block argument %a.
    // CHECK: dbg.variable "a", %a {typeName = "UInt"} : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "a", typeName: none = "UInt"> : () -> ()

    firrtl.connect %b, %a : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// 0-operand circt_debug_var resolving to a wire.

// CHECK-LABEL: firrtl.module @ZeroOperandVarWire
firrtl.circuit "ZeroOperandVarWire" {
  firrtl.module @ZeroOperandVarWire() {
    // CHECK: %mywire = firrtl.wire
    %mywire = firrtl.wire : !firrtl.uint<8>
    // CHECK: dbg.variable "mywire", %mywire {typeName = "UInt"} : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "mywire", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// 0-operand circt_debug_var with no matching port/wire/reg is treated as the
// memory case: the intrinsic is simply erased, no dbg.variable is produced.

// CHECK-LABEL: firrtl.module @ZeroOperandVarNoMatch
firrtl.circuit "ZeroOperandVarNoMatch" {
  firrtl.module @ZeroOperandVarNoMatch() {
    // CHECK-NOT: dbg.variable
    // CHECK-NOT: circt_debug_var
    firrtl.int.generic "circt_debug_var"
      <name: none = "nonexistent", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// 0-operand circt_debug_var resolving to a firrtl.reg.

// CHECK-LABEL: firrtl.module @ZeroOperandVarReg
firrtl.circuit "ZeroOperandVarReg" {
  firrtl.module @ZeroOperandVarReg(in %clk: !firrtl.clock) {
    // CHECK: %myreg = firrtl.reg
    %myreg = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK: dbg.variable "myreg", %myreg {typeName = "UInt"} : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "myreg", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// Nested aggregate: bundle whose field is itself a bundle. Exercises
// recursion in buildDebugAggregateWithMeta -- the outer struct contains
// an inner struct, with a leaf-meta entry on the deepest field.

// CHECK-LABEL: firrtl.module @NestedBundleTest
firrtl.circuit "NestedBundleTest" {
  firrtl.module @NestedBundleTest() {
    %io = firrtl.wire : !firrtl.bundle<inner: bundle<a: uint<4>, b: uint<4>>>

    // Subfield meta on a deeply nested leaf: io.inner.a
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.inner.a", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<inner: bundle<a: uint<4>, b: uint<4>>>) -> ()

    // CHECK:      firrtl.subfield %{{.*}}[inner]
    // CHECK:      firrtl.subfield %{{.*}}[a]
    // CHECK:      dbg.subfield "io.inner.a", %{{.*}} {typeName = "UInt"} : !firrtl.uint<4>
    // CHECK:      firrtl.subfield %{{.*}}[b]
    // CHECK:      dbg.struct
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io"
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "Outer">
      %io : (!firrtl.bundle<inner: bundle<a: uint<4>, b: uint<4>>>) -> ()
  }
}

// -----

// circt_debug_typedef is silently consumed: no representation in the IR yet,
// and no error.

// CHECK-LABEL: firrtl.module @TypeDefDropped
firrtl.circuit "TypeDefDropped" {
  firrtl.module @TypeDefDropped() {
    // CHECK-NOT: circt_debug_typedef
    firrtl.int.generic "circt_debug_typedef"
      <typeName: none = "MyAlias"> : () -> ()
  }
}
