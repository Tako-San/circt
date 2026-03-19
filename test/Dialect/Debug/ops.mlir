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
    // CHECK: %[[E0:.+]] = dbg.enumdef "MyState" fqn "pkg.MyState$" {Idle = 0 : i64, Run = 1 : i64}
    %e0 = dbg.enumdef "MyState" fqn "pkg.MyState$" {Idle = 0 : i64, Run = 1 : i64}

    // CHECK: arith.constant 0 : i2
    // CHECK-NEXT: dbg.variable "state", {{%.*}} enumDef %[[E0]] : i2
    %c = arith.constant 0 : i2
    dbg.variable "state", %c enumDef %e0 : i2

    // CHECK: = dbg.array [%
    %arr = dbg.array [%c, %c] : i2

    // CHECK: = dbg.struct
    %struct = dbg.struct {"a": %c, "b": %c} : i2, i2

    // CHECK: = dbg.scope "inner", "InnerMod"
    %scope = dbg.scope "inner", "InnerMod"

    // CHECK: = dbg.scope "nested", "Nested"
    %scope2 = dbg.scope "nested", "Nested" scope %scope

    // CHECK-NEXT: %[[E1:.+]] = dbg.enumdef "MyState" fqn "pkg.MyState$" {Idle = 0 : i64, Run = 1 : i64}
    %e1 = dbg.enumdef "MyState" fqn "pkg.MyState$" {Idle = 0 : i64, Run = 1 : i64}

    // CHECK: arith.constant 0 : i2
    // CHECK-NEXT: dbg.variable "state", %{{.*}} enumDef %[[E1]] {typeName = "MyState"} : i2
    %c2 = arith.constant 0 : i2
    dbg.variable "state", %c2 enumDef %e1 {typeName = "MyState"} : i2

    // CHECK: dbg.moduleinfo "MyClass", "MyInst", "MyClass.scala" : 10 params "{\22arg0\22:42}"
    dbg.moduleinfo "MyClass", "MyInst", "MyClass.scala" : 10 params "{\"arg0\":42}"

    // CHECK: dbg.meminfo "mem0", "SyncReadMem", 64, "Foo.scala" : 33, type "{\22className\22:\22UInt\22,\22width\22:16}"
    dbg.meminfo "mem0", "SyncReadMem", 64, "Foo.scala" : 33, type "{\"className\":\"UInt\",\"width\":16}"

    return
  }
}
