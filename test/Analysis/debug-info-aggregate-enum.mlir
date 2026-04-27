// RUN: circt-translate --dump-di --verify-diagnostics %s | FileCheck %s

// For aggregates, per-leaf enum info lives on the inner `dbg.subfield` ops;
// the DI analysis must not collapse them into the root `DIVariable::enumDef`
// slot (that would be lossy). Consumers that need the full mapping walk
// `DIVariable::value` themselves. `--verify-diagnostics` flags any stray
// warning from the analysis.

// CHECK-LABEL: Module "MixedLeafEnums" for hw.module
// CHECK: Variable "io"
hw.module @MixedLeafEnums(in %state: i2, in %mode: i3, in %data: i8) {
  %stateEnum = dbg.enumdef "State", fqn "pkg.State$", {Idle = 0 : i64, Run = 1 : i64}
  %modeEnum  = dbg.enumdef "Mode",  fqn "pkg.Mode$",  {Read = 0 : i64, Write = 1 : i64}

  // Two leaves with *different* enumDefs and one plain leaf.
  %s = dbg.subfield "io.state", %state enumDef %stateEnum : i2
  %m = dbg.subfield "io.mode",  %mode  enumDef %modeEnum  : i3
  %d = dbg.subfield "io.data",  %data                     : i8

  %agg = dbg.struct {"state": %s, "mode": %m, "data": %d} : !dbg.subfield, !dbg.subfield, !dbg.subfield

  // Root dbg.variable has no enumDef; previously this path triggered a
  // graph walk that warned "enum info for non-first leaves will be lost".
  // Now it's silent; --verify-diagnostics would catch any stray warning.
  dbg.variable "io", %agg : !dbg.struct<!dbg.subfield, !dbg.subfield, !dbg.subfield>
}
