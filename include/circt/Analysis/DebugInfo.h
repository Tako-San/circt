//===- DebugInfo.h - Debug info analysis ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_DEBUGINFO_H
#define CIRCT_ANALYSIS_DEBUGINFO_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

namespace circt {

struct DIInstance;
struct DIVariable;

namespace detail {
struct DebugInfoBuilder;
} // namespace detail

/// Compact per-module integer identifier for a `dbg.enumdef` op, used by
/// downstream exporters (e.g. HGLDD's `enum_def_ref`). Assigned by the DI
/// analysis when it first encounters each enumdef in a module.
using DIEnumDefId = uint32_t;

/// Map from an enum's raw integer value to its variant name.
using DIEnumValMap = SmallDenseMap<int64_t, StringAttr>;

/// Module-scoped table of all enum definitions: id -> variant map.
using DIEnumDefMap = SmallDenseMap<DIEnumDefId, DIEnumValMap>;

/// Source-language type information lifted off `dbg.moduleinfo`,
/// `dbg.variable`, and `dbg.subfield` ops. Mirrored into the HGLDD
/// `source_lang_type_info` field for consumption by waveform viewers.
struct DISourceLang {
  StringAttr typeName;
  ArrayAttr params;
};

struct DIModule {
  /// The operation that generated this level of hierarchy.
  Operation *op = nullptr;
  /// The name of this level of hierarchy.
  StringAttr name;
  /// Levels of hierarchy nested under this module.
  SmallVector<DIInstance *, 0> instances;
  /// Variables declared within this module.
  SmallVector<DIVariable *, 0> variables;
  /// If this is an extern declaration.
  bool isExtern = false;
  /// If this is an inline scope created by a `dbg.scope` operation.
  bool isInline = false;

  /// Source-language type, from `dbg.moduleinfo`.
  DISourceLang sourceLangType;

  /// Enum definitions visible at this module's scope.
  DIEnumDefMap enumDefinitions;

  /// Maps each `debug::EnumDefOp` in this module to its compact numeric ID
  /// (the one used as the key in `enumDefinitions`). Consumers that walk IR
  /// and need the same ID (e.g. when emitting `enum_def_ref` references from
  /// an arbitrary `dbg.enumdef` pointer) should look it up here instead of
  /// assigning their own counter.
  llvm::DenseMap<mlir::Operation *, DIEnumDefId> enumDefIds;
};

struct DIInstance {
  /// The operation that generated this instance.
  Operation *op = nullptr;
  /// The name of this instance.
  StringAttr name;
  /// The instantiated module.
  DIModule *module;
};

struct DIVariable {
  /// The name of this variable.
  StringAttr name;
  /// The location of the variable's declaration.
  LocationAttr loc;
  /// The SSA value representing the value of this variable.
  Value value = nullptr;
  /// Enum definition this variable refers to, if any. Populated only for
  /// scalar enum-typed variables (root `dbg.variable` has an `enumDef`
  /// operand). For aggregates, per-leaf enum refs live on inner `dbg.subfield`
  /// ops reachable via `value`; there is no lossy "pick one leaf" collapse.
  mlir::Value enumDef = nullptr;

  /// Source-language type, from `dbg.variable` attributes.
  DISourceLang sourceLangType;

  /// Reference to the enum definition by its per-module integer ID.
  /// Populated when `enumDef` is non-null and points at a `dbg.enumdef` op
  /// whose ID is registered in the enclosing `DIModule::enumDefinitions`
  /// table.
  std::optional<DIEnumDefId> enumDefRef = std::nullopt;
};

/// Debug information attached to an operation and the operations nested within.
///
/// This is an analysis that gathers debug information for a piece of IR, either
/// from attributes attached to operations or the general structure of the IR.
struct DebugInfo {
  /// Collect the debug information nested under the given operation.
  DebugInfo(Operation *op);

  /// The operation that was passed to the constructor.
  Operation *operation;
  /// A mapping from module name to module debug info.
  llvm::MapVector<StringAttr, DIModule *> moduleNodes;

protected:
  friend struct detail::DebugInfoBuilder;
  llvm::SpecificBumpPtrAllocator<DIModule> moduleAllocator;
  llvm::SpecificBumpPtrAllocator<DIInstance> instanceAllocator;
  llvm::SpecificBumpPtrAllocator<DIVariable> variableAllocator;
};

} // namespace circt

#endif // CIRCT_ANALYSIS_DEBUGINFO_H
