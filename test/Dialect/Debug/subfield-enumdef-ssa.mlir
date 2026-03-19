// RUN: circt-opt --verify-roundtrip %s | FileCheck %s

module {
  func.func @Test() {
    %e = dbg.enumdef "FieldEnum" fqn "pkg.FieldEnum" {VALUE = 42 : i64}
    %0 = arith.constant 0 : i32

    // CHECK: %[[SF:.+]] = dbg.subfield %{{.*}} enumDef %{{.*}} : i32
    %f = dbg.subfield %0 enumDef %e : i32

    // Используем в struct чтобы SubFieldOp не был dead-code
    %s = dbg.struct {"field": %f} : !dbg.subfield
    return
  }
}
