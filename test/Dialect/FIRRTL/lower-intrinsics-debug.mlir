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
