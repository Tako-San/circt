// RUN: circt-opt %s --verify-diagnostics --split-input-file

// -----
// expected-error @+1 {{expected 'fqn'}}
dbg.enumdef "Bad" "pkg.Bad$" {Idle = 0 : i64}

// -----
// expected-error @+1 {{'dbg.enumdef' op variant missing 'value' IntegerAttr}}
dbg.enumdef "Bad2" fqn "pkg.Bad2$" {Idle = "notanint"}
