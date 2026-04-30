//===- VerilogName.h - Resolve Verilog-side names for SSA values -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_VERILOGNAME_H
#define CIRCT_DIALECT_SV_VERILOGNAME_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Value.h"

namespace circt::sv {

/// Best-effort Verilog signal name for `value`: HW/SV module port name for
/// block-args; `hw.verilogName` (set by PrettifyVerilogNames) or `name` on
/// hw.wire / sv.wire / sv.reg / sv.logic; `hw.verilogName` on any other op;
/// the destination port name when the value flows into `hw.output`. Walks
/// through `sv.read_inout` so post-ExportVerilog spilled wires resolve.
/// Empty StringAttr if nothing applies.
inline mlir::StringAttr resolveVerilogName(mlir::Value value) {
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (auto mod =
            mlir::dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp()))
      return mod.getInputNameAttr(blockArg.getArgNumber());
    return {};
  }
  auto *op = mlir::cast<mlir::OpResult>(value).getOwner();
  auto pickName = [&](mlir::StringRef key) -> mlir::StringAttr {
    if (auto a = op->getAttrOfType<mlir::StringAttr>(key); a && !a.empty())
      return a;
    return {};
  };
  if (auto readInout = mlir::dyn_cast<sv::ReadInOutOp>(op))
    return resolveVerilogName(readInout.getInput());
  if (mlir::isa<hw::WireOp, sv::WireOp, sv::RegOp, sv::LogicOp>(op)) {
    if (auto a = pickName("hw.verilogName"))
      return a;
    if (auto b = pickName("name"))
      return b;
  }
  if (auto a = pickName("hw.verilogName"))
    return a;
  for (auto &use : op->getUses())
    if (auto out = mlir::dyn_cast<hw::OutputOp>(use.getOwner()))
      if (auto mod = out->getParentOfType<hw::HWModuleOp>())
        return mod.getOutputNameAttr(use.getOperandNumber());
  return {};
}

} // namespace circt::sv

#endif // CIRCT_DIALECT_SV_VERILOGNAME_H
