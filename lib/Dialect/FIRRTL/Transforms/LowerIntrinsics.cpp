//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// generic intrinsic operations and rewrites to their implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERINTRINSICS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass
    : public circt::firrtl::impl::LowerIntrinsicsBase<LowerIntrinsicsPass> {
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() override;

  std::shared_ptr<IntrinsicLowerings> lowering;
};
} // namespace

/// Build the immutable converter set once, shared across module invocations.
LogicalResult LowerIntrinsicsPass::initialize(MLIRContext *context) {
  IntrinsicLowerings local(context);
  IntrinsicLoweringInterfaceCollection collection(context);
  collection.populateIntrinsicLowerings(local);
  this->lowering = std::make_shared<IntrinsicLowerings>(std::move(local));
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  auto mod = getOperation();

  // The transient `firrtl.debug_leaves` attr is wiped on every exit path so
  // a phase-2 failure can't leak scaffolding into downstream passes.
  llvm::scope_exit cleanup{[&] { firrtl::clearDebugLeavesAttr(mod); }};

  // Phase 1: stage debug-intrinsic data (dbg.enumdef ops, firrtl.debug_leaves
  // attr) into the IR so converters don't depend on phase ordering.
  // (firrtl.module is a Graph region; block-start insertion is cosmetic.)
  OpBuilder builder = OpBuilder::atBlockBegin(mod.getBodyBlock());
  if (mlir::failed(firrtl::liftDebugIntrinsics(mod, builder)))
    return signalPassFailure();

  // Phase 2: run intrinsic lowerings.
  auto result = lowering->lower(mod);
  if (failed(result))
    return signalPassFailure();

  numConverted += *result;

  if (*result == 0)
    markAllAnalysesPreserved();
}
