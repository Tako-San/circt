//===- UhdiCaptureWhen.cpp - Record control flow into dbg.rootblock ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Capture each FIRRTL module's when/connect tree into a `dbg.rootblock`
// region before `firrtl-expand-whens` flattens it into muxes. String-based
// varRef / guardRef attrs survive the FIRRTL ops being rewritten.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-uhdi-capture-when"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_UHDICAPTUREWHEN
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {

struct GuardToken {
  std::string name;
  bool negated = false;
};

/// Walk up a chain of firrtl.subfield ops; returns true if the root is
/// a `firrtl.mem` op. Lets us pick the right separator convention when
/// naming the subfield path.
static bool isMemRootedSubfield(firrtl::SubfieldOp sub) {
  Operation *cur = sub.getInput().getDefiningOp();
  while (auto parentSub = dyn_cast_or_null<firrtl::SubfieldOp>(cur))
    cur = parentSub.getInput().getDefiningOp();
  return isa_and_nonnull<firrtl::MemOp>(cur);
}

/// Walk a FIRRTL XMR chain (`ref.resolve -> ref.sub* -> xmr.ref @sym`) and
/// resolve `@sym` to a `hw.hierpath`, joining its namepath with `.`. Empty
/// if the chain doesn't bottom out in an XMRRefOp or the symbol is missing
/// (caller falls back to `<complex>`). Note: hgdb's VPI evaluator may not
/// resolve such hierarchical paths; the breakpoint then degrades to
/// always-fire under the over-approximation policy.
static std::string xmrPathString(mlir::Value value) {
  Value cur = value;
  // Strip off the resolve, then any chain of ref.sub.
  if (auto resolve = cur.getDefiningOp<firrtl::RefResolveOp>())
    cur = resolve.getRef();
  while (auto sub = cur.getDefiningOp<firrtl::RefSubOp>())
    cur = sub.getInput();
  auto xmr = cur.getDefiningOp<firrtl::XMRRefOp>();
  if (!xmr)
    return {};
  auto hierPath = SymbolTable::lookupNearestSymbolFrom<hw::HierPathOp>(
      xmr, xmr.getRefAttr());
  if (!hierPath)
    return {};
  std::string out;
  for (auto entry : hierPath.getNamepath()) {
    if (!out.empty())
      out += '.';
    if (auto innerRef = dyn_cast<hw::InnerRefAttr>(entry))
      out += innerRef.getName().getValue().str();
    else if (auto sym = dyn_cast<FlatSymbolRefAttr>(entry))
      out += sym.getValue().str();
  }
  return out;
}

/// Best-effort source-level name for an SSA value.
/// Priority: 1) firrtl.module port arg; 2) named wire/reg/node;
/// 3) XMR chain rooted at `firrtl.xmr.ref` (joined with `.`);
/// 4) <parent>.<field> via a firrtl.subfield on a bundle; 5) any
/// consuming dbg.variable. Mem-rooted subfield chains use `_` instead
/// of `.` so the produced paths match post-LowerCHIRRTL flat-wire
/// naming (and native HGLDD's port_var names).
static std::string nameFor(mlir::Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto mod =
            dyn_cast<firrtl::FModuleOp>(blockArg.getOwner()->getParentOp()))
      return mod.getPortName(blockArg.getArgNumber()).str();
    return {};
  }
  if (auto opResult = dyn_cast<OpResult>(value)) {
    Operation *defOp = opResult.getOwner();
    if (auto name = defOp->getAttrOfType<StringAttr>("name");
        name && !name.getValue().empty())
      return name.getValue().str();
    // XMR chain: assemble the cross-module hierarchical path. Prevents
    // silent-drop on `connect dest, otherModule.signal` style sources;
    // see xmrPathString docs for the runtime-resolution caveat.
    if (isa<firrtl::RefResolveOp>(defOp))
      if (std::string path = xmrPathString(value); !path.empty())
        return path;
    if (auto sub = dyn_cast<firrtl::SubfieldOp>(defOp)) {
      std::string parent = nameFor(sub.getInput());
      if (parent.empty())
        return {};
      auto bundle = dyn_cast<firrtl::BundleType>(sub.getInput().getType());
      if (!bundle)
        return {};
      auto fname = bundle.getElementName(sub.getFieldIndex());
      if (fname.empty())
        return parent;
      const char *sep = isMemRootedSubfield(sub) ? "_" : ".";
      return parent + sep + fname.str();
    }
  }
  for (Operation *user : value.getUsers())
    if (auto var = dyn_cast<debug::VariableOp>(user))
      if (var.getValue() == value)
        return var.getName().str();
  return {};
}

/// Map a FIRRTL primop to its UHDI spec §5 opcode string. Empty when the
/// op isn't one we materialise into a `dbg.expression`.
static StringRef firrtlOpcode(Operation *op) {
  if (isa<firrtl::AndPrimOp>(op))
    return "&";
  if (isa<firrtl::OrPrimOp>(op))
    return "|";
  if (isa<firrtl::XorPrimOp>(op))
    return "^";
  if (isa<firrtl::NotPrimOp>(op))
    return "!";
  if (isa<firrtl::EQPrimOp>(op))
    return "==";
  if (isa<firrtl::NEQPrimOp>(op))
    return "!=";
  if (isa<firrtl::LTPrimOp>(op))
    return "<";
  if (isa<firrtl::LEQPrimOp>(op))
    return "<=";
  if (isa<firrtl::GTPrimOp>(op))
    return ">";
  if (isa<firrtl::GEQPrimOp>(op))
    return ">=";
  return "";
}

/// Look up an SSA value at module-body scope that carries the same
/// runtime value as `value`. Returns `value` itself if it's already at
/// module body (block-arg or top-level op result), or if a `dbg.variable`
/// wrapping it lives there. Null otherwise; caller breaks the chain.
static mlir::Value findModuleBodyProxy(mlir::Value value, FModuleOp module) {
  Block *moduleBody = module.getBodyBlock();
  // Direct: the value itself is at module body.
  Block *owner = isa<BlockArgument>(value)
                     ? cast<BlockArgument>(value).getOwner()
                     : cast<OpResult>(value).getOwner()->getBlock();
  if (owner == moduleBody)
    return value;
  // Search for a dbg.variable consuming this value whose own scope is the
  // module body.
  for (Operation *user : value.getUsers())
    if (auto var = dyn_cast<debug::VariableOp>(user))
      if (var.getValue() == value && var->getBlock() == moduleBody)
        return value; // dbg.variable annotates it; safe to use as-is.
  return {};
}

/// Materialise a `dbg.expression` tree at module-body level for a compound
/// when-condition. Each operand is either an existing module-body proxy or
/// the result of a recursive call. Returns null on unsupported shapes
/// (caller falls back to `<complex>` sentinel).
static mlir::Value materializeExpression(mlir::Value cond,
                                         debug::RootBlockOp rootBlock,
                                         FModuleOp module, unsigned &counter) {
  auto opResult = dyn_cast<OpResult>(cond);
  if (!opResult)
    return {};
  Operation *defOp = opResult.getOwner();
  StringRef opcode = firrtlOpcode(defOp);
  if (opcode.empty())
    return {};

  // Resolve each operand to a dominating handle.
  SmallVector<Value, 2> operands;
  for (Value operand : defOp->getOperands()) {
    if (Value proxy = findModuleBodyProxy(operand, module)) {
      operands.push_back(proxy);
      continue;
    }
    // Recurse: maybe operand is a nested compound primop.
    Value sub = materializeExpression(operand, rootBlock, module, counter);
    if (!sub)
      return {};
    operands.push_back(sub);
  }

  auto *ctx = rootBlock.getContext();
  std::string name =
      ("__uhdi_expr_" + module.getName() + "_" + std::to_string(counter++))
          .str();
  OpBuilder b(rootBlock);
  auto expr = debug::ExpressionOp::create(
      b, module.getLoc(),
      debug::ExpressionType::get(ctx), // result type
      StringAttr::get(ctx, name), StringAttr::get(ctx, opcode), operands,
      /*scope=*/Value());
  return expr.getResult();
}

/// Materialise + return the synthesised name (for guardRef StringAttr).
/// Empty if materialisation failed.
static std::string materializeExpressionName(mlir::Value cond,
                                             debug::RootBlockOp rootBlock,
                                             FModuleOp module,
                                             unsigned &counter) {
  Value result = materializeExpression(cond, rootBlock, module, counter);
  if (!result)
    return {};
  auto *defOp = result.getDefiningOp();
  if (auto expr = dyn_cast<debug::ExpressionOp>(defOp))
    return expr.getName().str();
  return {};
}

/// Synthesize `dbg.variable "<mem>_<port>_<field>"` per `firrtl.mem`
/// port-field. Names match post-LowerCHIRRTL flat-wire / native HGLDD
/// convention so statement-tree refs and the snapshot pass agree.
static void synthesizeMemPortVariables(FModuleOp module) {
  module.walk([&](firrtl::MemOp memOp) {
    StringRef memName = memOp.getName();
    OpBuilder b(module.getContext());
    b.setInsertionPointAfter(memOp);
    for (size_t i = 0, e = memOp->getNumResults(); i < e; ++i) {
      Value port = memOp->getResult(i);
      StringRef portName = memOp.getPortName(i);
      auto bundleType = dyn_cast<firrtl::BundleType>(port.getType());
      if (!bundleType)
        continue;
      for (size_t f = 0, fe = bundleType.getNumElements(); f < fe; ++f) {
        StringRef fieldName = bundleType.getElementName(f);
        auto subfield = firrtl::SubfieldOp::create(b, memOp.getLoc(), port, f);
        std::string varName =
            (memName + "_" + portName + "_" + fieldName).str();
        debug::VariableOp::create(b, memOp.getLoc(), b.getStringAttr(varName),
                                  subfield.getResult(),
                                  /*typeName=*/StringAttr(),
                                  /*params=*/ArrayAttr(),
                                  /*enumDef=*/Value(),
                                  /*scope=*/Value());
      }
    }
  });
}

/// `name1[&[!]name2...]`; `<complex>` placeholder for unresolvable guards.
/// Downstream uhdi-to-hgdb re-parses this back into an SV expression.
static std::string serializeGuardStack(ArrayRef<GuardToken> stack) {
  std::string out;
  for (auto &tok : stack) {
    if (!out.empty())
      out += '&';
    if (tok.negated)
      out += '!';
    out += tok.name.empty() ? std::string("<complex>") : tok.name;
  }
  return out;
}

static void emitConnectStmt(firrtl::FConnectLike connect, OpBuilder &b,
                            ArrayRef<GuardToken> stack) {
  std::string dest = nameFor(connect.getDest());
  std::string src = nameFor(connect.getSrc());
  if (dest.empty()) {
    // Anonymous LHS -- can't represent as an assignment to a
    // named signal.  Drop and log under -debug-only.
    LLVM_DEBUG(llvm::dbgs()
               << "uhdi: drop connect at " << connect.getLoc()
               << " (dest='', src='" << src << "')\n");
    return;
  }
  if (src.empty()) {
    // Constant-driven (`connect r, UInt<8>(0)`) or driven by a
    // temp-wire / subexpression result (`connect r, tail(...)`):
    // there's no source-level dbg.variable to name.  Synthesise
    // a placeholder so the connect still appears in the body --
    // downstream hgdb-firrtl emits these as proper assignments
    // and skipping them here means our debug.db loses a row per
    // such connect (visible as `assignment` table divergence in
    // bench).
    src = "<const>";
  }
  // If the connect targets a regreset, the effective guard is
  // `!<reset> && <existing user guards>` -- the reset has implicit
  // priority and any user-written `r := ...` only fires when the
  // reset signal is low.  hgdb-firrtl synthesises this implicit
  // !reset by walking the FIRRTL graph; we make it explicit on
  // the UHDI side so the .uhdi document carries the same semantic
  // information and downstream converters reproduce the same
  // breakpoint condition.
  SmallVector<GuardToken> effectiveStack;
  if (auto regreset = dyn_cast_or_null<firrtl::RegResetOp>(
          connect.getDest().getDefiningOp())) {
    std::string resetName = nameFor(regreset.getResetSignal());
    if (!resetName.empty())
      effectiveStack.push_back({resetName, /*negated=*/true});
  }
  effectiveStack.append(stack.begin(), stack.end());

  auto *ctx = b.getContext();
  NamedAttrList bp;
  if (std::string e = serializeGuardStack(effectiveStack); !e.empty())
    bp.set("enableRef", StringAttr::get(ctx, e));
  debug::ConnectStmtOp::create(b, connect.getLoc(), StringAttr::get(ctx, dest),
                               StringAttr::get(ctx, src),
                               bp.empty() ? DictionaryAttr{}
                                          : DictionaryAttr::get(ctx, bp));
}

static void emitDeclStmt(Operation *decl, OpBuilder &b) {
  // wire / reg / regreset / node carry `$name`.
  StringAttr name = decl->getAttrOfType<StringAttr>("name");
  if (!name || name.getValue().empty())
    return;
  debug::DeclStmtOp::create(b, decl->getLoc(), name);
}

/// Walk state: insertion anchor for materialised `dbg.expression` ops
/// and a per-module counter for their synthesised names.
struct ExprSink {
  debug::RootBlockOp rootBlock;
  FModuleOp module;
  unsigned counter = 0;
};

static void walkRegion(Region &region, OpBuilder &b,
                       SmallVectorImpl<GuardToken> &stack, ExprSink &sink) {
  if (region.empty())
    return;
  auto *ctx = b.getContext();
  auto emitBranch = [&](Region &branch, Location loc, std::string &&guardName,
                        bool negated) {
    auto block = debug::SubBlockOp::create(
        b, loc,
        StringAttr::get(ctx, guardName.empty() ? "<complex>" : guardName),
        BoolAttr::get(ctx, negated));
    block.getBody().emplaceBlock();
    OpBuilder inner = OpBuilder::atBlockBegin(&block.getBody().front());
    stack.push_back({guardName, negated});
    walkRegion(branch, inner, stack, sink);
    stack.pop_back();
  };
  for (Operation &op : region.front()) {
    if (auto when = dyn_cast<firrtl::WhenOp>(op)) {
      std::string guard = nameFor(when.getCondition());
      if (guard.empty())
        // Compound guard: materialise a dbg.expression; empty result falls
        // back to "<complex>" sentinel downstream.
        guard = materializeExpressionName(when.getCondition(), sink.rootBlock,
                                          sink.module, sink.counter);
      emitBranch(when.getThenRegion(), when.getLoc(), std::string(guard),
                 /*negated=*/false);
      if (when.hasElseRegion() && !when.getElseRegion().empty())
        emitBranch(when.getElseRegion(), when.getLoc(), std::move(guard),
                   /*negated=*/true);
    } else if (auto connect = dyn_cast<firrtl::FConnectLike>(op)) {
      emitConnectStmt(connect, b, stack);
    } else if (isa<firrtl::WireOp, firrtl::RegOp, firrtl::RegResetOp,
                   firrtl::NodeOp>(op)) {
      emitDeclStmt(&op, b);
    }
  }
}

struct UhdiCaptureWhenPass
    : public circt::firrtl::impl::UhdiCaptureWhenBase<UhdiCaptureWhenPass> {
  void runOnOperation() override;
};

} // namespace

void UhdiCaptureWhenPass::runOnOperation() {
  firrtl::FModuleOp module = getOperation();

  // Idempotent on re-runs.
  bool already = false;
  module.walk([&](debug::RootBlockOp) {
    already = true;
    return WalkResult::interrupt();
  });
  if (already)
    return;

  // Mem-port wrappers must exist before the walk so `nameFor` resolves
  // `_`-joined mem-rooted subfields against them.
  synthesizeMemPortVariables(module);

  OpBuilder top = OpBuilder::atBlockEnd(module.getBodyBlock());
  auto body = debug::RootBlockOp::create(top, module.getLoc());
  body.getBody().emplaceBlock();
  OpBuilder inner = OpBuilder::atBlockBegin(&body.getBody().front());

  SmallVector<GuardToken, 4> stack;
  ExprSink sink{body, module, /*counter=*/0};
  walkRegion(*module.getBodyBlock()->getParent(), inner, stack, sink);
}
