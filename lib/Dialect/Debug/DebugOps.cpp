//===- DebugOps.cpp - Debug dialect operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
// EnumDefOp
//===----------------------------------------------------------------------===//
LogicalResult EnumDefOp::verify() {
  for (auto attr : getVariants()) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return emitOpError("each element of 'variants' must be a DictionaryAttr");
    if (!dict.get("name") || !isa<StringAttr>(dict.get("name")))
      return emitOpError("variant missing 'name' StringAttr");
    if (!dict.get("value") || !isa<IntegerAttr>(dict.get("value")))
      return emitOpError("variant missing 'value' IntegerAttr");
  }
  return success();
}

ParseResult EnumDefOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *ctx = parser.getContext();

  StringAttr nameAttr, fqnAttr;
  if (parser.parseAttribute(nameAttr, "name", result.attributes))
    return failure();
  if (parser.parseKeyword("fqn") || parser.parseAttribute(fqnAttr))
    return failure();
  result.addAttribute("fqn", fqnAttr);

  SmallVector<Attribute> variants;
  auto parseVariant = [&]() -> ParseResult {
    StringRef varName;
    if (parser.parseKeyword(&varName) || parser.parseEqual())
      return failure();
    Attribute valueAttr;
    if (parser.parseAttribute(valueAttr))
      return failure();
    NamedAttribute entries[] = {
        {StringAttr::get(ctx, "name"), StringAttr::get(ctx, varName)},
        {StringAttr::get(ctx, "value"), valueAttr}};
    variants.push_back(DictionaryAttr::get(ctx, entries));
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces,
                                     parseVariant))
    return failure();

  // Add variants before parseOptionalAttrDict to avoid duplication
  result.addAttribute("variants", ArrayAttr::get(ctx, variants));
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  result.addTypes(EnumDefType::get(parser.getContext()));
  return success();
}

void EnumDefOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttribute(getNameAttr());
  p << " fqn ";
  p.printAttribute(getFqnAttr());
  p << " {";
  llvm::interleaveComma(getVariants(), p.getStream(), [&](Attribute attr) {
    auto dict = cast<DictionaryAttr>(attr);
    p << cast<StringAttr>(dict.get("name")).getValue();
    p << " = ";
    p.printAttribute(dict.get("value"));
  });
  p << '}';
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     {"name", "fqn", "variants"});
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns for EnumDefOp
//===----------------------------------------------------------------------===//

namespace {
struct DeduplicateEnumDef : public OpRewritePattern<debug::EnumDefOp> {
  using OpRewritePattern<debug::EnumDefOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(debug::EnumDefOp op,
                                PatternRewriter &rewriter) const override {
    // Look for another EnumDefOp earlier in the same block with same fqn
    Block *block = op->getBlock();
    for (auto &sibling : *block) {
      auto other = dyn_cast<debug::EnumDefOp>(&sibling);
      if (!other || other == op)
        continue;
      if (other.getFqn() == op.getFqn()) {
        // Replace all uses of current op with earlier definition
        rewriter.replaceOp(op, other.getResult());
        return success();
      }
    }
    return failure();
  }
};
} // namespace

void EnumDefOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *ctx) {
  results.add<DeduplicateEnumDef>(ctx);
}

// Operation implementations generated from `Debug.td`
#define GET_OP_CLASSES
#include "circt/Dialect/Debug/Debug.cpp.inc"

void DebugDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Debug/Debug.cpp.inc"
      >();
}
