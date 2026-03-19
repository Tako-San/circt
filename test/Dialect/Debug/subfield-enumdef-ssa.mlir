// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

module {
  func.func @Test() {
    %e = dbg.enumdef "FieldEnum" fqn "pkg.FieldEnum" {VALUE = 42 : i64}
    %0 = arith.constant 0 : i32
    %f = dbg.subfield value, %0 enumDef %e : i32
    return
  }
}
// CHECK: %{{.+}} = dbg.enumdef
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %{{[0-9]+}} = dbg.subfield "value", %c0_i32 enumDef %{{[0-9]+}} : i32
