// RUN: circt-opt %s | FileCheck %s

module {
  func.func @Test() {
    %e = dbg.enumdef "MyEnum" fqn "pkg.MyEnum" {RED = 0 : i64, GREEN = 1 : i64}
    %0 = arith.constant 1 : i32
    // CHECK: %[[E:.+]] = dbg.enumdef
    // CHECK: dbg.variable {{.+}} enumDef %[[E]]
    dbg.variable "color", %0 enumDef %e : i32
    return
  }
}
