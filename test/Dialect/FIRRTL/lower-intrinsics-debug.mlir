// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s | FileCheck %s

// CHECK-LABEL: firrtl.module @DebugTest
firrtl.circuit "DebugTest" {
  firrtl.module @DebugTest(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {

    // 1. moduleinfo
    // CHECK: dbg.moduleinfo "MyModule", "DebugTest", "MyModule.scala" : 7
    firrtl.int.generic "circt_debug_moduleinfo"
      <className: none = "MyModule", name: none = "DebugTest",
       sourceFile: none = "MyModule.scala", sourceLine: si64 = 7>
      : () -> ()

    // 2. enumdef with JSON array
    // CHECK: dbg.enumdef "MyState" fqn "pkg.MyState$" {Idle = 0 : i64, Run = 1 : i64}
    firrtl.int.generic "circt_debug_enumdef"
      <name: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":0,\"valueStr\":\"0\"},{\"name\":\"Run\",\"value\":1,\"valueStr\":\"1\"}]">
      : () -> ()

    // 3. typetag on scalar wire → dbg.variable with the wire value
    // CHECK: dbg.variable "w", %{{.*}} {typeName = "UInt"} : !firrtl.uint<8>
    %w = firrtl.wire : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_typetag"
      <name: none = "w", className: none = "UInt", width: si64 = 8,
       binding: none = "wire", direction: none = "unspecified",
       sourceFile: none = "MyModule.scala", sourceLine: si64 = 9>
      %w : (!firrtl.uint<8>) -> ()

    // 4. typetag on a passive bundle → dbg.struct + dbg.variable
    //    The converter builds:
    //      %a = firrtl.subfield %io[in]
    //      %b = firrtl.subfield %io[out]
    //      %s = dbg.struct {"in": %a, "out": %b}
    //      dbg.variable "io", %s
    // CHECK:      firrtl.subfield %io[in]
    // CHECK:      firrtl.subfield %io[out]
    // CHECK:      dbg.struct
    // CHECK-NEXT: dbg.variable "io", %{{.*}} {typeName = "MyBundle"}
    %io = firrtl.wire : !firrtl.bundle<in: uint<8>, out: uint<8>>
    firrtl.int.generic "circt_debug_typetag"
      <name: none = "io", className: none = "MyBundle", width: si64 = -1,
       binding: none = "wire", direction: none = "unspecified",
       sourceFile: none = "MyModule.scala", sourceLine: si64 = 20>
      %io : (!firrtl.bundle<in: uint<8>, out: uint<8>>) -> ()

    // 5. typetag on a Vec → dbg.array + dbg.variable
    // CHECK:      firrtl.subindex %v[0]
    // CHECK:      firrtl.subindex %v[1]
    // CHECK:      dbg.array
    // CHECK-NEXT: dbg.variable "v", %{{.*}} {typeName = "Vec"}
    %v = firrtl.wire : !firrtl.vector<uint<4>, 2>
    firrtl.int.generic "circt_debug_typetag"
      <name: none = "v", className: none = "Vec", width: si64 = -1,
       binding: none = "wire", direction: none = "unspecified",
       sourceFile: none = "MyModule.scala", sourceLine: si64 = 22>
      %v : (!firrtl.vector<uint<4>, 2>) -> ()

    // 6. meminfo → dbg.meminfo
    // CHECK: dbg.meminfo "m", "SyncReadMem", 32, "MyModule.scala" : 12, type
    firrtl.int.generic "circt_debug_meminfo"
      <memName: none = "m", memoryKind: none = "SyncReadMem", depth: si64 = 32,
       sourceFile: none = "MyModule.scala", sourceLine: si64 = 12,
       dataType: none = "{\"className\":\"UInt\",\"width\":8}">
      : () -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// 7. Bundle с полем-enum — поле должно получить dbg.subfield с enumDef
//
// Chisel эмитирует:
//   circt_debug_enumdef для MyState
//   circt_debug_typetag(parent="io") для io.state — лист с enumTypeFqn
//   circt_debug_typetag(parent="")  для io       — корень
//
// Ожидаем:
//   dbg.enumdef "MyState"
//   firrtl.subfield %io_bundle[state]
//   dbg.subfield %state_val enumDef %e   ← обёртка только для enum-поля
//   firrtl.subfield %io_bundle[data]     ← data без обёртки
//   dbg.struct {"state": %subfield_result, "data": %data_val}
//   dbg.variable "io", %struct {typeName = "MyBundle"}

// CHECK-LABEL: firrtl.module @BundleEnumFieldTest
firrtl.circuit "BundleEnumFieldTest" {
  firrtl.module @BundleEnumFieldTest(
      in %in: !firrtl.uint<8>,
      out %out: !firrtl.uint<8>) {

    // enumdef — должен появиться в выводе
    // CHECK: %[[E:.+]] = dbg.enumdef "MyState" fqn "pkg.MyState$"
    firrtl.int.generic "circt_debug_enumdef"
      <name: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":0},{\"name\":\"Run\",\"value\":1}]">
      : () -> ()

    %io_bundle = firrtl.wire : !firrtl.bundle<state: uint<2>, data: uint<8>>

    // Лист io.state с enumTypeFqn — НЕ должен создавать dbg.variable
    // CHECK-NOT: dbg.variable "io.state"
    firrtl.int.generic "circt_debug_typetag"
      <name: none = "io.state", className: none = "UInt", width: si64 = 2,
       binding: none = "wire", direction: none = "unspecified",
       sourceFile: none = "Test.scala", sourceLine: si64 = 5,
       parent: none = "io",
       enumType: none = "MyState", enumTypeFqn: none = "pkg.MyState$">
      %io_bundle : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    // Лист io.data без enum — тоже НЕ создаёт dbg.variable
    // CHECK-NOT: dbg.variable "io.data"
    firrtl.int.generic "circt_debug_typetag"
      <name: none = "io.data", className: none = "UInt", width: si64 = 8,
       binding: none = "wire", direction: none = "unspecified",
       sourceFile: none = "Test.scala", sourceLine: si64 = 6,
       parent: none = "io">
      %io_bundle : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    // Корневой typetag — создаёт dbg.struct + dbg.variable
    // CHECK:      firrtl.subfield %{{.*}}[state]
    // CHECK:      dbg.subfield %{{.*}} enumDef %[[E]] : !firrtl.uint<2>
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK-NOT:  dbg.subfield %{{.*}} : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io"
    firrtl.int.generic "circt_debug_typetag"
      <name: none = "io", className: none = "MyBundle", width: si64 = -1,
       binding: none = "wire", direction: none = "unspecified",
       sourceFile: none = "Test.scala", sourceLine: si64 = 4>
      %io_bundle : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}
