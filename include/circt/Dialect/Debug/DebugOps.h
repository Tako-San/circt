//===- DebugOps.h - Debug dialect operations ====----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGOPS_H
#define CIRCT_DIALECT_DEBUG_DEBUGOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/Debug/DebugTypes.h"

// Operation definitions generated from `Debug.td`
#define GET_OP_CLASSES
#include "circt/Dialect/Debug/Debug.h.inc"

namespace circt {
namespace debug {

/// UHDI attr names stamped on dbg.* ops by uhdi-init / verilog-snapshot
/// and consumed by EmitUHDI.
inline constexpr llvm::StringLiteral kUhdiStableIdAttr = "uhdi.stable_id";
inline constexpr llvm::StringLiteral kUhdiReprEntryAttr = "uhdi.repr_entry";
inline constexpr llvm::StringLiteral kUhdiVerilogRepr = "verilog";

/// Walk every `dbg.rootblock` within `root` and diagnose statement-tree
/// references (varRef / valueRef / guardRef) that don't resolve to a
/// `dbg.variable` in the enclosing module. Returns the number of
/// diagnostics emitted; 0 means every ref was resolvable.
///
/// Not hooked into the default MLIR verifier because literal-string
/// fallback is an *intentional* feature of the statement tree (covers
/// mem-port subfield paths, XMR references, and synthesized names
/// capture-when produces before their corresponding `dbg.variable` ops
/// exist). Call explicitly from tests or a lint-style CLI when a
/// stricter check is desired.
unsigned verifyUhdiStatementRefs(mlir::Operation *root);

} // namespace debug
} // namespace circt

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_H
