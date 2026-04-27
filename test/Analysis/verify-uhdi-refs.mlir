// RUN: circt-opt --test-verify-uhdi-refs %s --verify-diagnostics

// Statement-tree refs that DO resolve to a `dbg.variable` are silent.
hw.module @Resolved(in %a : i1, in %b : i1) {
  dbg.variable "a", %a : i1
  dbg.variable "b", %b : i1
  dbg.rootblock {
    dbg.decl_stmt "a"
    dbg.subblock guard "a" {
      dbg.connect_stmt "b" = "a"
    }
  }
}

// `<complex>` is the well-known sentinel for a guard that capture-when
// couldn't reduce to a single name. The verifier expects to see it as a
// guardRef on real-world inputs and must NOT fire on it.
hw.module @ComplexGuardSentinel(in %a : i1) {
  dbg.variable "a", %a : i1
  dbg.rootblock {
    dbg.subblock guard "<complex>" {
      dbg.connect_stmt "a" = "a"
    }
  }
}

// `dbg.expression` names also count as resolvable -- capture-when
// materialises one for compound when-guards and the body's guardRef
// then points at it by synthesised name.
hw.module @ExpressionGuard(in %en : i1, in %valid : i1, in %x : i1) {
  %0 = comb.and %en, %valid : i1
  dbg.variable "en", %en : i1
  dbg.variable "valid", %valid : i1
  dbg.variable "x", %x : i1
  %expr = dbg.expression "g_and", opcode "&", operands(%en, %valid : i1, i1)
  dbg.rootblock {
    dbg.subblock guard "g_and" {
      dbg.connect_stmt "x" = "en"
    }
  }
}

// Refs that don't resolve to any dbg.variable in the enclosing module
// trigger an explicit warning per ref. Literal-string fallback is the
// emitter's runtime behaviour; this lint helper just makes the gap loud.
hw.module @Unresolved(in %a : i1) {
  dbg.variable "a", %a : i1
  dbg.rootblock {
    // expected-warning @below {{uhdi: statement guardRef 'ghost'}}
    dbg.subblock guard "ghost" {
      // expected-warning @below {{uhdi: statement varRef 'phantom'}}
      // expected-warning @below {{uhdi: statement valueRef 'spirit'}}
      dbg.connect_stmt "phantom" = "spirit"
    }
    // expected-warning @below {{uhdi: statement varRef 'wraith'}}
    dbg.decl_stmt "wraith"
  }
}
