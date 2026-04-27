//===- DebugOps.cpp - Debug dialect operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringSet.h"

using namespace circt;
using namespace debug;
using namespace mlir;

//===----------------------------------------------------------------------===//
// StructOp
//===----------------------------------------------------------------------===//

ParseResult StructOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the struct fields.
  SmallVector<Attribute> names;
  SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  std::string nameBuffer;
  auto parseField = [&]() {
    nameBuffer.clear();
    if (parser.parseString(&nameBuffer) || parser.parseColon() ||
        parser.parseOperand(operands.emplace_back()))
      return failure();
    names.push_back(StringAttr::get(parser.getContext(), nameBuffer));
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces, parseField))
    return failure();

  // Parse the attribute dictionary.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse the field types, if there are any fields.
  SmallVector<Type> types;
  if (!operands.empty()) {
    if (parser.parseColon())
      return failure();
    auto typesLoc = parser.getCurrentLocation();
    if (parser.parseTypeList(types))
      return failure();
    if (types.size() != operands.size())
      return parser.emitError(typesLoc,
                              "number of fields and types must match");
  }

  // Resolve the operands.
  for (auto [operand, type] : llvm::zip(operands, types))
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();

  // Finalize the op.
  result.addAttribute("names", ArrayAttr::get(parser.getContext(), names));
  result.addTypes(StructType::get(parser.getContext()));
  return success();
}

void StructOp::print(OpAsmPrinter &printer) {
  printer << " {";
  llvm::interleaveComma(llvm::zip(getFields(), getNames()), printer.getStream(),
                        [&](auto pair) {
                          auto [field, name] = pair;
                          printer.printAttribute(name);
                          printer << ": ";
                          printer.printOperand(field);
                        });
  printer << '}';
  printer.printOptionalAttrDict(getOperation()->getAttrs(), {"names"});
  if (!getFields().empty()) {
    printer << " : ";
    printer << getFields().getTypes();
  }
}

//===----------------------------------------------------------------------===//
// ArrayOp
//===----------------------------------------------------------------------===//

ParseResult ArrayOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the elements, attributes and types.
  SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  if (parser.parseOperandList(operands, AsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Resolve the operands.
  if (!operands.empty()) {
    Type type;
    if (parser.parseColon() || parser.parseType(type))
      return failure();
    for (auto operand : operands)
      if (parser.resolveOperand(operand, type, result.operands))
        return failure();
  }

  // Finalize the op.
  result.addTypes(ArrayType::get(parser.getContext()));
  return success();
}

void ArrayOp::print(OpAsmPrinter &printer) {
  printer << " [";
  printer << getElements();
  printer << ']';
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  if (!getElements().empty()) {
    printer << " : ";
    printer << getElements()[0].getType();
  }
}

//===----------------------------------------------------------------------===//
// Generated operation code
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "circt/Dialect/Debug/Debug.cpp.inc"

void DebugDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Debug/Debug.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// EnumDefOp canonicalization
//===----------------------------------------------------------------------===//

namespace {
struct EnumDefDeduplication : public OpRewritePattern<EnumDefOp> {
  using OpRewritePattern<EnumDefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(EnumDefOp op,
                                PatternRewriter &rewriter) const override {
    // Equivalence requires fqn + variants + scope. Scope is load-bearing:
    // same-fqn defs in different inline scopes share a source type but their
    // result SSA tokens must not be merged.
    auto *block = op->getBlock();
    auto opFqn = op.getFqn();
    auto opVariants = op.getVariantsMap();
    auto opScope = op.getScope();

    for (auto &otherOp : *block) {
      if (&otherOp == op.getOperation())
        break;
      auto otherEnumDef = dyn_cast<EnumDefOp>(otherOp);
      if (!otherEnumDef)
        continue;

      if (otherEnumDef.getFqn() == opFqn &&
          otherEnumDef.getVariantsMap() == opVariants &&
          otherEnumDef.getScope() == opScope) {
        rewriter.replaceOp(op, otherEnumDef.getResult());
        return success();
      }
    }
    return failure();
  }
};
} // namespace

void EnumDefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<EnumDefDeduplication>(context);
}

//===----------------------------------------------------------------------===//
// UHDI statement-tree reference check
//===----------------------------------------------------------------------===//

unsigned debug::verifyUhdiStatementRefs(Operation *root) {
  unsigned diagnostics = 0;

  // Walk each enclosing module (hw.module / firrtl.module / ...) that owns a
  // dbg.rootblock; collect the set of dbg.variable names it declares, then
  // walk the rootblock's statements and diagnose any unresolved refs.
  root->walk([&](RootBlockOp rootBlock) {
    Operation *enclosingModule = rootBlock->getParentOp();
    if (!enclosingModule)
      return;

    llvm::StringSet<> knownNames;
    enclosingModule->walk(
        [&](VariableOp var) { knownNames.insert(var.getName()); });
    // `dbg.expression` is a sibling value-tracker (used as the synthesised
    // guardRef target for compound when-conditions); count its names too so
    // a body's guardRef into a materialised expression doesn't fire a false
    // positive.
    enclosingModule->walk(
        [&](ExpressionOp expr) { knownNames.insert(expr.getName()); });

    auto checkRef =
        [&](Operation *stmt, StringRef refKind, StringRef name) {
          // `<complex>` is the well-known sentinel that capture-when emits when
          // a guard expression can't be reduced to a single dbg.variable /
          // dbg.expression name; treat it as expected, not a defect.
          if (name.empty() || name == "<complex>" || knownNames.contains(name))
            return;
          stmt->emitWarning()
              << "uhdi: statement " << refKind << " '" << name
              << "' has no matching dbg.variable in the enclosing module; "
                 "the emitter will fall back to the literal name";
          ++diagnostics;
        };

    rootBlock.walk([&](Operation *op) {
      if (auto c = dyn_cast<ConnectStmtOp>(op)) {
        checkRef(op, "varRef", c.getVarRef());
        checkRef(op, "valueRef", c.getValueRef());
      } else if (auto b = dyn_cast<SubBlockOp>(op)) {
        checkRef(op, "guardRef", b.getGuardRef());
      } else if (auto d = dyn_cast<DeclStmtOp>(op)) {
        checkRef(op, "varRef", d.getVarRef());
      }
    });
  });

  return diagnostics;
}
