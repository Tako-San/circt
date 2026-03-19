// RUN: circt-opt --canonicalize %s | FileCheck %s

module {
  func.func @Test() {
    // CHECK: %[[E:.+]] = dbg.enumdef "S" fqn "p.S"
    // CHECK-NOT: = dbg.enumdef "S" fqn "p.S"
    %e0 = dbg.enumdef "S" fqn "p.S" {A = 0 : i64}
    %e1 = dbg.enumdef "S" fqn "p.S" {A = 0 : i64}

    %0 = arith.constant 0 : i2
    // CHECK: dbg.variable "x", {{.*}} enumDef %[[E]]
    // CHECK: dbg.variable "y", {{.*}} enumDef %[[E]]
    dbg.variable "x", %0 enumDef %e0 : i2
    dbg.variable "y", %0 enumDef %e1 : i2
    return
  }
}
