//===- UhdiVerilogSnapshot.cpp - Snapshot Verilog names onto dbg.* ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For each dbg.* op with a `uhdi.stable_id`, attach a `uhdi.repr_entry`
// dictionary with the Verilog-side name. EmitUHDI reads this to populate
// each variable's `representations.verilog.name`.
//
// Schedule: after PrettifyVerilogNames (so `hw.verilogName` is final) and
// before ExportVerilog / EmitUHDI.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/SV/VerilogName.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_UHDIVERILOGSNAPSHOT
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

/// `#uhdi.repr_entry<verilog = {name = "..."}>`. Null if no name.
static DictionaryAttr buildReprEntry(MLIRContext *ctx, StringAttr name) {
  if (!name || name.empty())
    return {};
  auto perRepr = DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "name"), name)});
  return DictionaryAttr::get(
      ctx,
      {NamedAttribute(StringAttr::get(ctx, debug::kUhdiVerilogRepr), perRepr)});
}

struct UhdiVerilogSnapshotPass
    : public circt::hw::impl::UhdiVerilogSnapshotBase<UhdiVerilogSnapshotPass> {
  void runOnOperation() override;
};

} // namespace

void UhdiVerilogSnapshotPass::runOnOperation() {
  hw::HWModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  StringAttr idAttr = StringAttr::get(ctx, debug::kUhdiStableIdAttr);
  StringAttr reprAttr = StringAttr::get(ctx, debug::kUhdiReprEntryAttr);

  // Index every hw.instance once: source-level name -> Verilog-side name
  // (hw.verilogName if present, source-level otherwise). One walk replaces
  // the previous per-dbg.scope rescan.
  llvm::DenseMap<StringAttr, StringAttr> instVerilogName;
  module.walk([&](hw::InstanceOp inst) {
    auto src = inst.getInstanceNameAttr();
    auto vname = inst->getAttrOfType<StringAttr>("hw.verilogName");
    instVerilogName.try_emplace(src, vname ? vname : src);
  });

  module.walk([&](Operation *op) {
    // Only stamp the ops firrtl-uhdi-init owns; idempotent on re-runs.
    // dbg.struct/array don't have a single Verilog-level name (the
    // emitter walks into their operands per-field).
    if (!op->hasAttr(idAttr) || op->hasAttr(reprAttr))
      return;
    StringAttr name;
    if (auto var = dyn_cast<debug::VariableOp>(op)) {
      name = sv::resolveVerilogName(var.getValue());
    } else if (auto scope = dyn_cast<debug::ScopeOp>(op)) {
      auto it = instVerilogName.find(scope.getInstanceNameAttr());
      if (it != instVerilogName.end())
        name = it->second;
    } else {
      return;
    }
    if (auto entry = buildReprEntry(ctx, name))
      op->setAttr(reprAttr, entry);
  });
}
