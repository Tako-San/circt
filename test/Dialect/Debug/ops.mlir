// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: module {
module {
  // CHECK-LABEL: func.func @ModuleInfoTest
  func.func @ModuleInfoTest() -> () {
    // CHECK: dbg.moduleinfo "MyMod", "InstName", "MyMod.scala" : 42
    dbg.moduleinfo "MyMod", "InstName", "MyMod.scala" : 42
    return
  }

  // CHECK-LABEL: func.func @EnumTest
  func.func @EnumTest() -> () {
    // CHECK: dbg.enumdef "MyState" fqn "pkg.MyState$" {Idle = 0 : i64, Run = 1 : i64}
    %e = dbg.enumdef "MyState" fqn "pkg.MyState$" {Idle = 0 : i64, Run = 1 : i64}

    // CHECK: arith.constant 0 : i2
    // CHECK-NEXT: dbg.variable "state", {{%.*}} {enumRef = "MyState"} : i2
    %c = arith.constant 0 : i2
    dbg.variable "state", %c {enumRef = "MyState"} : i2

    // CHECK: = dbg.array [%
    %arr = dbg.array [%c, %c] : i2

    // CHECK: = dbg.struct
    %struct = dbg.struct {"a": %c, "b": %c} : i2, i2

    // CHECK: = dbg.scope "inner", "InnerMod"
    %scope = dbg.scope "inner", "InnerMod"

    // CHECK: = dbg.scope "nested", "Nested"
    %scope2 = dbg.scope "nested", "Nested" scope %scope

    return
  }
}