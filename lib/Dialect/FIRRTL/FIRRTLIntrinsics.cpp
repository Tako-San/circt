//===- FIRRTLIntrinsics.cpp - Lower Intrinsics ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Support/JSON.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;

// vtable anchor
void IntrinsicConverter::anchor() {}

//===----------------------------------------------------------------------===//
// GenericIntrinsic
//===----------------------------------------------------------------------===//

// Checks for a number of operands between n and n+c (allows for c optional
// inputs)
ParseResult GenericIntrinsic::hasNInputs(unsigned n, unsigned c) {
  auto numOps = op.getNumOperands();
  unsigned m = n + c;
  if (numOps < n || numOps > m) {
    auto err = emitError() << " has " << numOps << " inputs instead of ";
    if (c == 0)
      err << n;
    else
      err << " between " << n << " and " << m;
    return failure();
  }
  return success();
}

// Accessor method for the number of inputs
unsigned GenericIntrinsic::getNumInputs() { return op.getNumOperands(); }

ParseResult GenericIntrinsic::hasNOutputElements(unsigned n) {
  auto b = getOutputBundle();
  if (!b)
    return emitError() << " missing output bundle";
  if (b.getType().getNumElements() != n)
    return emitError() << " has " << b.getType().getNumElements()
                       << " output elements instead of " << n;
  return success();
}

ParseResult GenericIntrinsic::hasNParam(unsigned n, unsigned c) {
  unsigned num = 0;
  if (op.getParameters())
    num = op.getParameters().size();
  if (num < n || num > n + c) {
    auto d = emitError() << " has " << num << " parameters instead of ";
    if (c == 0)
      d << n;
    else
      d << " between " << n << " and " << (n + c);
    return failure();
  }
  return success();
}

ParseResult GenericIntrinsic::namedParam(StringRef paramName, bool optional) {
  for (auto a : op.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue() == paramName) {
      if (isa<StringAttr>(param.getValue()))
        return success();

      return emitError() << " has parameter '" << param.getName()
                         << "' which should be a string but is not";
    }
  }
  if (optional)
    return success();
  return emitError() << " is missing parameter " << paramName;
}

ParseResult GenericIntrinsic::namedIntParam(StringRef paramName,
                                            bool optional) {
  for (auto a : op.getParameters()) {
    auto param = cast<ParamDeclAttr>(a);
    if (param.getName().getValue() == paramName) {
      if (isa<IntegerAttr>(param.getValue()))
        return success();

      return emitError() << " has parameter '" << param.getName()
                         << "' which should be an integer but is not";
    }
  }
  if (optional)
    return success();
  return emitError() << " is missing parameter " << paramName;
}

//===----------------------------------------------------------------------===//
// IntrinsicOpConversion
//===----------------------------------------------------------------------===//

/// Conversion pattern adaptor dispatching via generic intrinsic name.
namespace {
class IntrinsicOpConversion final
    : public OpConversionPattern<GenericIntrinsicOp> {
public:
  using ConversionMapTy = IntrinsicLowerings::ConversionMapTy;

  IntrinsicOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                        const ConversionMapTy &conversions,
                        size_t &numConversions,
                        bool allowUnknownIntrinsics = false)
      : OpConversionPattern(typeConverter, context), conversions(conversions),
        numConversions(numConversions),
        allowUnknownIntrinsics(allowUnknownIntrinsics) {}

  LogicalResult
  matchAndRewrite(GenericIntrinsicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto it = conversions.find(op.getIntrinsicAttr());
    if (it == conversions.end()) {
      if (!allowUnknownIntrinsics)
        return op.emitError("unknown intrinsic ") << op.getIntrinsicAttr();
      return failure();
    }

    auto &conv = *it->second;
    auto result = conv.checkAndConvert(GenericIntrinsic(op), adaptor, rewriter);
    if (succeeded(result))
      ++numConversions;
    return result;
  }

private:
  const ConversionMapTy &conversions;
  size_t &numConversions;
  const bool allowUnknownIntrinsics;
};
} // namespace

//===----------------------------------------------------------------------===//
// IntrinsicLowerings
//===----------------------------------------------------------------------===//

FailureOr<size_t> IntrinsicLowerings::lower(FModuleOp mod,
                                            bool allowUnknownIntrinsics) {

  ConversionTarget target(*context);

  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  if (allowUnknownIntrinsics)
    target.addDynamicallyLegalOp<GenericIntrinsicOp>(
        [this](GenericIntrinsicOp op) {
          return !conversions.contains(op.getIntrinsicAttr());
        });
  else
    target.addIllegalOp<GenericIntrinsicOp>();

  // Automatically insert wires + connect for compatible FIRRTL base types.
  // For now, this is not customizable/extendable.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

  auto firrtlBaseTypeMaterialization =
      [](OpBuilder &builder, FIRRTLBaseType resultType, ValueRange inputs,
         Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    auto inputType = type_dyn_cast<FIRRTLBaseType>(inputs.front().getType());
    if (!inputType)
      return {};

    if (!areTypesEquivalent(resultType, inputType) ||
        !isTypeLarger(resultType, inputType))
      return {};

    auto w = WireOp::create(builder, loc, resultType).getResult();
    emitConnect(builder, loc, w, inputs.front());
    return w;
  };
  // New result doesn't match? Add wire + connect.
  typeConverter.addSourceMaterialization(firrtlBaseTypeMaterialization);
  // New operand doesn't match? Add wire + connect.
  typeConverter.addTargetMaterialization(firrtlBaseTypeMaterialization);

  RewritePatternSet patterns(context);
  size_t count = 0;
  patterns.add<IntrinsicOpConversion>(typeConverter, context, conversions,
                                      count, allowUnknownIntrinsics);

  if (failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
    return failure();

  return count;
}

//===----------------------------------------------------------------------===//
// IntrinsicLoweringInterfaceCollection
//===----------------------------------------------------------------------===//

void IntrinsicLoweringInterfaceCollection::populateIntrinsicLowerings(
    IntrinsicLowerings &lowering) const {
  for (const IntrinsicLoweringDialectInterface &interface : *this)
    interface.populateIntrinsicLowerings(lowering);
}

//===----------------------------------------------------------------------===//
// FIRRTL intrinsic lowering converters
//===----------------------------------------------------------------------===//

namespace {

class CirctSizeofConverter : public IntrinsicOpConverter<SizeOfIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedOutput<UIntType>(32) || gi.hasNParam(0);
  }
};

class CirctIsXConverter : public IntrinsicOpConverter<IsXIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedOutput<UIntType>(1) || gi.hasNParam(0);
  }
};

class CirctPlusArgTestConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(0) || gi.sizedOutput<UIntType>(1) ||
           gi.namedParam("FORMAT") || gi.hasNParam(1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    rewriter.replaceOpWithNewOp<PlusArgsTestIntrinsicOp>(
        gi.op, gi.getParamValue<StringAttr>("FORMAT"));
  }
};

class CirctPlusArgValueConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNOutputElements(2) ||
           gi.sizedOutputElement<UIntType>(0, "found", 1) ||
           gi.hasOutputElement(1, "result") || gi.namedParam("FORMAT") ||
           gi.hasNParam(1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto bty = gi.getOutputBundle().getType();
    auto newop = PlusArgsValueIntrinsicOp::create(
        rewriter, gi.op.getLoc(), bty.getElementTypePreservingConst(0),
        bty.getElementTypePreservingConst(1),
        gi.getParamValue<StringAttr>("FORMAT"));
    rewriter.replaceOpWithNewOp<BundleCreateOp>(
        gi.op, bty, ValueRange({newop.getFound(), newop.getResult()}));
  }
};

class CirctClockGateConverter
    : public IntrinsicOpConverter<ClockGateIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    if (gi.op.getNumOperands() == 3) {
      return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
             gi.sizedInput<UIntType>(2, 1) || gi.typedOutput<ClockType>() ||
             gi.hasNParam(0);
    }
    if (gi.op.getNumOperands() == 2) {
      return gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
             gi.typedOutput<ClockType>() || gi.hasNParam(0);
    }
    gi.emitError() << " has " << gi.op.getNumOperands()
                   << " ports instead of 3 or 4";
    return true;
  }
};

class CirctClockInverterConverter
    : public IntrinsicOpConverter<ClockInverterIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.typedInput<ClockType>(0) ||
           gi.typedOutput<ClockType>() || gi.hasNParam(0);
  }
};

class CirctClockDividerConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.typedInput<ClockType>(0) ||
           gi.typedOutput<ClockType>() || gi.namedIntParam("POW_2") ||
           gi.hasNParam(1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto pow2 =
        gi.getParamValue<IntegerAttr>("POW_2").getValue().getZExtValue();

    auto pow2Attr = rewriter.getI64IntegerAttr(pow2);

    rewriter.replaceOpWithNewOp<ClockDividerIntrinsicOp>(
        gi.op, adaptor.getOperands()[0], pow2Attr);
  }
};

template <typename OpTy>
class CirctLTLBinaryConverter : public IntrinsicOpConverter<OpTy> {
public:
  using IntrinsicOpConverter<OpTy>::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedInput<UIntType>(1, 1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

template <typename OpTy>
class CirctLTLUnaryConverter : public IntrinsicOpConverter<OpTy> {
public:
  using IntrinsicOpConverter<OpTy>::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.hasNParam(0);
  }
};

class CirctLTLDelayConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("delay") ||
           gi.namedIntParam("length", true) || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto delay = getI64Attr(gi.getParamValue<IntegerAttr>("delay"));
    auto length = getI64Attr(gi.getParamValue<IntegerAttr>("length"));
    rewriter.replaceOpWithNewOp<LTLDelayIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], delay, length);
  }
};

class CirctLTLPastConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    if (gi.hasNInputs(1, 1) || gi.sizedInput<UIntType>(0, 1) ||
        gi.sizedOutput<UIntType>(1) || gi.namedIntParam("delay") ||
        gi.hasNParam(1))
      return true;
    if (gi.op.getNumOperands() > 1 && gi.typedInput<ClockType>(1))
      return true;
    return false;
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto delay = rewriter.getI64IntegerAttr(
        gi.getParamValue<IntegerAttr>("delay").getValue().getZExtValue());
    auto operands = adaptor.getOperands();
    Value clock;
    if (operands.size() > 1)
      clock = operands[1];
    rewriter.replaceOpWithNewOp<LTLPastIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), operands[0], delay, clock);
  }
};

class CirctLTLClockConverter
    : public IntrinsicOpConverter<LTLClockIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.typedInput<ClockType>(1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

class CirctLTLRepeatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("base") ||
           gi.namedIntParam("more", true) || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto base = getI64Attr(gi.getParamValue<IntegerAttr>("base"));
    auto more = getI64Attr(gi.getParamValue<IntegerAttr>("more"));
    rewriter.replaceOpWithNewOp<LTLRepeatIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], base, more);
  }
};

class CirctLTLGoToRepeatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("base") ||
           gi.namedIntParam("more") || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto base = getI64Attr(gi.getParamValue<IntegerAttr>("base"));
    auto more = getI64Attr(gi.getParamValue<IntegerAttr>("more"));
    rewriter.replaceOpWithNewOp<LTLGoToRepeatIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], base, more);
  }
};

class CirctLTLNonConsecutiveRepeatConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1) || gi.sizedInput<UIntType>(0, 1) ||
           gi.sizedOutput<UIntType>(1) || gi.namedIntParam("base") ||
           gi.namedIntParam("more") || gi.hasNParam(1, 1);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto getI64Attr = [&](IntegerAttr val) {
      if (!val)
        return IntegerAttr();
      return rewriter.getI64IntegerAttr(val.getValue().getZExtValue());
    };
    auto base = getI64Attr(gi.getParamValue<IntegerAttr>("base"));
    auto more = getI64Attr(gi.getParamValue<IntegerAttr>("more"));
    rewriter.replaceOpWithNewOp<LTLNonConsecutiveRepeatIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), adaptor.getOperands()[0], base, more);
  }
};

template <class Op>
class CirctVerifConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(1, 2) || gi.sizedInput<UIntType>(0, 1) ||
           gi.namedParam("label", true) || gi.hasNParam(0, 1) ||
           gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto label = gi.getParamValue<StringAttr>("label");
    auto operands = adaptor.getOperands();

    // Check if an enable was provided
    Value enable;
    if (gi.getNumInputs() == 2)
      enable = operands[1];

    rewriter.replaceOpWithNewOp<Op>(gi.op, operands[0], enable, label);
  }
};

class CirctMux2CellConverter : public IntrinsicConverter {
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(3) || gi.typedInput<UIntType>(0) || gi.hasNParam(0) ||
           gi.hasOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<Mux2CellIntrinsicOp>(gi.op, operands[0],
                                                     operands[1], operands[2]);
  }
};

class CirctMux4CellConverter : public IntrinsicConverter {
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(5) || gi.typedInput<UIntType>(0) || gi.hasNParam(0) ||
           gi.hasOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<Mux4CellIntrinsicOp>(
        gi.op, operands[0], operands[1], operands[2], operands[3], operands[4]);
  }
};

class CirctHasBeenResetConverter
    : public IntrinsicOpConverter<HasBeenResetIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.typedInput<ClockType>(0) ||
           gi.hasResetInput(1) || gi.sizedOutput<UIntType>(1) ||
           gi.hasNParam(0);
  }
};

class CirctProbeConverter : public IntrinsicOpConverter<FPGAProbeIntrinsicOp> {
public:
  using IntrinsicOpConverter::IntrinsicOpConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(2) || gi.typedInput<ClockType>(1) || gi.hasNParam(0) ||
           gi.hasNoOutput();
  }
};

template <class OpTy, bool ifElseFatal = false>
class CirctAssertConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  LogicalResult checkAndConvert(GenericIntrinsic gi,
                                GenericIntrinsicOpAdaptor adaptor,
                                PatternRewriter &rewriter) override {
    // Check structure of the intrinsic.
    if (gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
        gi.sizedInput<UIntType>(2, 1) ||
        gi.namedParam("format", /*optional=*/true) ||
        gi.namedParam("label", /*optional=*/true) ||
        gi.namedParam("guards", /*optional=*/true) || gi.hasNParam(0, 3) ||
        gi.hasNoOutput())
      return failure();

    auto format = gi.getParamValue<StringAttr>("format");
    auto label = gi.getParamValue<StringAttr>("label");
    auto guards = gi.getParamValue<StringAttr>("guards");

    auto clock = adaptor.getOperands()[0];
    auto predicate = adaptor.getOperands()[1];
    auto enable = adaptor.getOperands()[2];

    auto substitutions = adaptor.getOperands().drop_front(3);
    auto name = label ? label.strref() : "";

    // Parse the format string to handle special substitutions like
    // {{SimulationTime}} and {{HierarchicalModuleName}}
    StringAttr message;
    SmallVector<Value> allOperands;
    if (format) {
      SmallVector<Value> substitutionVec(substitutions.begin(),
                                         substitutions.end());
      if (failed(parseFormatString(rewriter, gi.op->getLoc(), format.getValue(),
                                   substitutionVec, message, allOperands)))
        return failure();
    } else {
      // Message is not optional, so provide empty string if not present.
      message = rewriter.getStringAttr("");
      allOperands.append(substitutions.begin(), substitutions.end());
    }

    auto op = rewriter.template replaceOpWithNewOp<OpTy>(
        gi.op, clock, predicate, enable, message, allOperands, name,
        /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      guards.strref().split(guardStrings, ';', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
      rewriter.startOpModification(op);
      op->setAttr("guards", rewriter.getStrArrayAttr(guardStrings));
      rewriter.finalizeOpModification(op);
    }

    if constexpr (ifElseFatal) {
      rewriter.startOpModification(op);
      op->setAttr("format", rewriter.getStringAttr("ifElseFatal"));
      rewriter.finalizeOpModification(op);
    }

    return success();
  }
};

class CirctCoverConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(3) || gi.hasNoOutput() ||
           gi.typedInput<ClockType>(0) || gi.sizedInput<UIntType>(1, 1) ||
           gi.sizedInput<UIntType>(2, 1) ||
           gi.namedParam("label", /*optional=*/true) ||
           gi.namedParam("guards", /*optional=*/true) || gi.hasNParam(0, 2);
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto label = gi.getParamValue<StringAttr>("label");
    auto guards = gi.getParamValue<StringAttr>("guards");

    auto clock = adaptor.getOperands()[0];
    auto predicate = adaptor.getOperands()[1];
    auto enable = adaptor.getOperands()[2];

    auto name = label ? label.strref() : "";
    // Empty message string for cover, only 'name' / label.
    auto message = rewriter.getStringAttr("");
    auto op = rewriter.replaceOpWithNewOp<CoverOp>(
        gi.op, clock, predicate, enable, message, ValueRange{}, name,
        /*isConcurrent=*/true);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      guards.strref().split(guardStrings, ';', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
      rewriter.startOpModification(op);
      op->setAttr("guards", rewriter.getStrArrayAttr(guardStrings));
      rewriter.finalizeOpModification(op);
    }
  }
};

class CirctUnclockedAssumeConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.sizedInput<UIntType>(0, 1) || gi.sizedInput<UIntType>(1, 1) ||
           gi.namedParam("format", /*optional=*/true) ||
           gi.namedParam("label", /*optional=*/true) ||
           gi.namedParam("guards", /*optional=*/true) || gi.hasNParam(0, 3) ||
           gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto format = gi.getParamValue<StringAttr>("format");
    auto label = gi.getParamValue<StringAttr>("label");
    auto guards = gi.getParamValue<StringAttr>("guards");

    auto predicate = adaptor.getOperands()[0];
    auto enable = adaptor.getOperands()[1];

    auto substitutions = adaptor.getOperands().drop_front(2);
    auto name = label ? label.strref() : "";
    // Message is not optional, so provide empty string if not present.
    auto message = format ? format : rewriter.getStringAttr("");
    auto op = rewriter.template replaceOpWithNewOp<UnclockedAssumeIntrinsicOp>(
        gi.op, predicate, enable, message, substitutions, name);
    if (guards) {
      SmallVector<StringRef> guardStrings;
      guards.strref().split(guardStrings, ';', /*MaxSplit=*/-1,
                            /*KeepEmpty=*/false);
      rewriter.startOpModification(op);
      op->setAttr("guards", rewriter.getStrArrayAttr(guardStrings));
      rewriter.finalizeOpModification(op);
    }
  }
};

class CirctDPICallConverter : public IntrinsicConverter {
  static bool getIsClocked(GenericIntrinsic gi) {
    return !gi.getParamValue<IntegerAttr>("isClocked").getValue().isZero();
  }

public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    if (gi.hasNParam(2, 2) || gi.namedIntParam("isClocked") ||
        gi.namedParam("functionName") ||
        gi.namedParam("inputNames", /*optional=*/true) ||
        gi.namedParam("outputName", /*optional=*/true))
      return true;
    auto isClocked = getIsClocked(gi);
    // If clocked, the first operand must be a clock.
    if (isClocked && gi.typedInput<ClockType>(0))
      return true;
    // Enable must be UInt<1>.
    if (gi.sizedInput<UIntType>(isClocked, 1))
      return true;

    return false;
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto isClocked = getIsClocked(gi);
    auto functionName = gi.getParamValue<StringAttr>("functionName");
    ArrayAttr inputNamesStrArray;
    StringAttr outputStr = gi.getParamValue<StringAttr>("outputName");
    if (auto inputNames = gi.getParamValue<StringAttr>("inputNames")) {
      SmallVector<StringRef> inputNamesTemporary;
      inputNames.strref().split(inputNamesTemporary, ';', /*MaxSplit=*/-1,
                                /*KeepEmpty=*/false);
      inputNamesStrArray = rewriter.getStrArrayAttr(inputNamesTemporary);
    }
    // Clock and enable are optional.
    Value clock = isClocked ? adaptor.getOperands()[0] : Value();
    Value enable = adaptor.getOperands()[static_cast<size_t>(isClocked)];

    auto inputs =
        adaptor.getOperands().drop_front(static_cast<size_t>(isClocked) + 1);

    rewriter.replaceOpWithNewOp<DPICallIntrinsicOp>(
        gi.op, gi.op.getResultTypes(), functionName, inputNamesStrArray,
        outputStr, clock, enable, inputs);
  }
};

//===----------------------------------------------------------------------===//
// Debug intrinsic converters
//===----------------------------------------------------------------------===//

static ArrayAttr parseParamsJSON(MLIRContext *ctx, StringAttr paramsStr,
                                 Operation *warnAt);

class CirctDebugModuleInfoConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    return gi.hasNInputs(0) || gi.namedParam("typeName") ||
           gi.namedParam("params", /*optional=*/true) || gi.hasNParam(1, 1) ||
           gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto typeName = gi.getParamValue<StringAttr>("typeName");
    auto paramsStr = gi.getParamValue<StringAttr>("params");

    ArrayAttr params;
    if (paramsStr) {
      params = parseParamsJSON(rewriter.getContext(), paramsStr, gi.op);
    }

    rewriter.replaceOpWithNewOp<debug::ModuleInfoOp>(gi.op, typeName, params);
  }
};

//===----------------------------------------------------------------------===//
// Debug intrinsic helpers
//===----------------------------------------------------------------------===//

/// Parse a JSON array `[{"name":..., "value":...}, ...]` into an ArrayAttr
/// of DictionaryAttrs, preserving native JSON typing for values. Returns null
/// on null/empty input. Parse failures are emitted as warnings on `warnAt`
/// (if non-null) and yield null.
static ArrayAttr parseParamsJSON(MLIRContext *ctx, StringAttr paramsStr,
                                 Operation *warnAt) {
  if (!paramsStr || paramsStr.getValue().empty())
    return {};

  auto parsed = llvm::json::parse(paramsStr.strref());
  if (auto err = parsed.takeError()) {
    auto msg = llvm::toString(std::move(err));
    if (warnAt)
      warnAt->emitWarning()
          << "debug params JSON failed to parse (type-parameter info "
             "dropped): "
          << msg;
    return {};
  }

  auto *arr = parsed->getAsArray();
  if (!arr) {
    if (warnAt)
      warnAt->emitWarning()
          << "debug params JSON is not a JSON array (type-parameter info "
             "dropped)";
    return {};
  }

  SmallVector<Attribute> entries;
  for (const auto &item : *arr) {
    auto *obj = item.getAsObject();
    if (!obj)
      continue;
    SmallVector<NamedAttribute> fields;
    for (auto &[k, v] : *obj) {
      auto key = StringAttr::get(ctx, StringRef(k));
      // Preserve native JSON typing so downstream consumers can dispatch on
      // the attribute type (e.g. BoolAttr vs. StringAttr "true").
      if (auto b = v.getAsBoolean())
        fields.push_back({key, BoolAttr::get(ctx, *b)});
      else if (auto s = v.getAsString())
        fields.push_back({key, StringAttr::get(ctx, *s)});
      else if (auto i = v.getAsInteger())
        fields.push_back(
            {key, IntegerAttr::get(IntegerType::get(ctx, 64), *i)});
      // Floats and nested objects are not emitted by the current frontend;
      // skip silently.
    }
    entries.push_back(DictionaryAttr::get(ctx, fields));
  }
  return ArrayAttr::get(ctx, entries);
}

/// Per-leaf metadata collected from the module's `firrtl.debug_leaves`
/// attribute. Keyed by the subfield's full dotted path (e.g. "io.in.a").
namespace {
struct LeafMeta {
  Value enumDefVal;
  StringAttr typeName;
  ArrayAttr params;
};
} // namespace
using LeafMetaMap = llvm::StringMap<LeafMeta>;

static Value buildDebugAggregateWithMeta(OpBuilder &builder, Location loc,
                                         Value value, StringRef parentPath,
                                         const LeafMetaMap &leafMetaMap,
                                         bool isRoot = true) {
  auto wrapIfMeta = [&](Value inner) -> Value {
    if (isRoot)
      return inner;
    auto it = leafMetaMap.find(parentPath);
    if (it == leafMetaMap.end())
      return inner;
    const auto &meta = it->second;
    auto nameAttr = StringAttr::get(builder.getContext(), parentPath);
    return debug::SubFieldOp::create(builder, loc, nameAttr, inner,
                                     meta.typeName, meta.params,
                                     meta.enumDefVal)
        .getResult();
  };
  return FIRRTLTypeSwitch<Type, Value>(value.getType())
      .Case<BundleType>([&](BundleType type) -> Value {
        SmallVector<Value> fields;
        SmallVector<Attribute> names;
        SmallVector<Operation *> subOps;
        for (auto [index, element] : llvm::enumerate(type.getElements())) {
          auto subOp = SubfieldOp::create(builder, loc, value, index);
          subOps.push_back(subOp.getOperation());
          std::string fieldPath =
              (Twine(parentPath) + "." + element.name.getValue()).str();
          if (auto dbgVal = buildDebugAggregateWithMeta(
                  builder, loc, subOp.getResult(), StringRef(fieldPath),
                  leafMetaMap, /*isRoot=*/false)) {
            fields.push_back(dbgVal);
            names.push_back(element.name);
          }
        }
        Value result = debug::StructOp::create(builder, loc, fields,
                                               builder.getArrayAttr(names));
        for (auto *sub : subOps)
          if (sub->use_empty())
            sub->erase();
        return wrapIfMeta(result);
      })
      .Case<FVectorType>([&](FVectorType type) -> Value {
        SmallVector<Value> elements;
        SmallVector<Operation *> subOps;
        for (unsigned i = 0; i < type.getNumElements(); ++i) {
          auto subOp = SubindexOp::create(builder, loc, value, i);
          subOps.push_back(subOp.getOperation());
          std::string elemPath =
              (Twine(parentPath) + "[" + Twine(i) + "]").str();
          if (auto dbgVal = buildDebugAggregateWithMeta(
                  builder, loc, subOp.getResult(), StringRef(elemPath),
                  leafMetaMap, /*isRoot=*/false))
            elements.push_back(dbgVal);
        }
        Value result;
        if (!elements.empty() && elements.size() == type.getNumElements())
          result = debug::ArrayOp::create(builder, loc, elements);
        for (auto *sub : subOps)
          if (sub->use_empty())
            sub->erase();
        return wrapIfMeta(result);
      })
      .Case<FIRRTLBaseType>([&](FIRRTLBaseType type) -> Value {
        if (!type.isGround())
          return {};
        return wrapIfMeta(value);
      })
      .Default([](auto) -> Value { return {}; });
}

/// Converts `circt_debug_var` to `dbg.variable`, reading per-module state
/// (module-level `dbg.enumdef` ops and `firrtl.debug_leaves`) staged by the
/// pre-passes in `LowerIntrinsicsPass`.
class CirctDebugVarConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;

  bool check(GenericIntrinsic gi) override {
    // 0 operands for memories (no SSA), 1 operand for normal values.
    unsigned n = gi.op.getNumOperands();
    if (n != 0 && n != 1)
      return true;
    return gi.namedParam("name") || gi.namedParam("typeName") ||
           gi.namedParam("params", /*optional=*/true) ||
           gi.namedParam("enumFqn", /*optional=*/true) ||
           gi.hasNParam(2, 3) || gi.hasNoOutput();
  }

  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor adaptor,
               PatternRewriter &rewriter) override {
    auto varName = gi.getParamValue<StringAttr>("name");
    auto typeName = gi.getParamValue<StringAttr>("typeName");
    auto location = gi.op.getLoc();
    auto modOp = gi.op->getParentOfType<FModuleOp>();

    ArrayAttr params;
    if (auto paramAttr = gi.getParamValue<StringAttr>("params"))
      params = parseParamsJSON(rewriter.getContext(), paramAttr, gi.op);

    // Index module-level dbg.enumdef ops once for both the var and its leaves.
    llvm::StringMap<Value> enumDefByFqn;
    if (modOp)
      modOp.walk([&](debug::EnumDefOp edOp) {
        enumDefByFqn.try_emplace(edOp.getFqn(), edOp.getResult());
      });
    auto lookupEnumDef = [&](StringAttr fqnAttr) -> Value {
      if (!fqnAttr || fqnAttr.getValue().empty())
        return {};
      auto it = enumDefByFqn.find(fqnAttr.getValue());
      return it == enumDefByFqn.end() ? Value{} : it->second;
    };

    Value enumDef = lookupEnumDef(gi.getParamValue<StringAttr>("enumFqn"));

    // Pick leaves belonging to this var from `firrtl.debug_leaves`, keyed by
    // their display path (matches what buildDebugAggregateWithMeta reconstructs).
    LeafMetaMap leafMap;
    StringRef varPath = varName ? varName.getValue() : StringRef{};
    if (modOp && !varPath.empty()) {
      auto leavesAttr = modOp->getAttrOfType<ArrayAttr>("firrtl.debug_leaves");
      if (leavesAttr) {
        for (auto entry : leavesAttr.getAsRange<DictionaryAttr>()) {
          auto parentAttr = entry.getAs<StringAttr>("parent");
          if (!parentAttr || parentAttr.getValue() != varPath)
            continue;
          auto pathAttr = entry.getAs<StringAttr>("name");
          if (!pathAttr)
            continue;
          LeafMeta meta;
          meta.typeName = entry.getAs<StringAttr>("typeName");
          meta.params = entry.getAs<ArrayAttr>("params");
          meta.enumDefVal = lookupEnumDef(entry.getAs<StringAttr>("enumFqn"));
          leafMap[pathAttr.getValue()] = meta;
        }
      }
    }

    // Locate the SSA to walk: direct operand, or (for non-passive roots
    // emitted with 0 operands) a port/wire/reg named `varName`.
    Value rawSignal;
    if (!adaptor.getOperands().empty()) {
      rawSignal = adaptor.getOperands()[0];
    } else if (modOp && varName) {
      StringRef wanted = varName.getValue();
      for (auto [port, arg] :
           llvm::zip(modOp.getPorts(), modOp.getArguments())) {
        if (port.name.getValue() == wanted) {
          rawSignal = arg;
          break;
        }
      }
      if (!rawSignal) {
        modOp.walk([&](Operation *op) {
          if (rawSignal)
            return;
          if (auto *namedOp = TypeSwitch<Operation *, Operation *>(op)
                                  .Case<WireOp, NodeOp, RegOp, RegResetOp>(
                                      [&](auto o) -> Operation * {
                                        return o.getNameAttr().getValue() ==
                                                       wanted
                                                   ? op
                                                   : nullptr;
                                      })
                                  .Default(nullptr))
            rawSignal = namedOp->getResult(0);
        });
      }
    }

    if (!rawSignal) {
      // Memory case (or unresolved): just erase the intrinsic.
      rewriter.eraseOp(gi.op);
      return;
    }

    Value dbgValue = buildDebugAggregateWithMeta(rewriter, location, rawSignal,
                                                 varPath, leafMap);
    if (!dbgValue)
      dbgValue = rawSignal;

    rewriter.replaceOpWithNewOp<debug::VariableOp>(gi.op, varName, dbgValue,
                                                   typeName, params, enumDef,
                                                   /*scope=*/Value{});
  }
};

/// Placeholder: silently consumes `circt_debug_typedef` until the pipeline
/// grows a representation for source-language typedefs.
class CirctDebugTypeDefConverter : public IntrinsicConverter {
public:
  using IntrinsicConverter::IntrinsicConverter;
  bool check(GenericIntrinsic gi) override { return false; }
  void convert(GenericIntrinsic gi, GenericIntrinsicOpAdaptor,
               PatternRewriter &rewriter) override {
    rewriter.eraseOp(gi.op);
  }
};

//===----------------------------------------------------------------------===//
// View intrinsic converter and helpers
//===----------------------------------------------------------------------===//

template <typename A>
A tryGetAs(DictionaryAttr dict, Attribute root, StringRef key, Location loc,
           Twine path = Twine()) {
  return tryGetAsBase<A>(dict, root, key, loc, "View 'info'",
                         "'info' attribute", path);
}

/// Recursively walk a sifive.enterprise.grandcentral.AugmentedType to extract
/// and slightly restructure information needed for a view.
std::optional<DictionaryAttr>
parseAugmentedType(MLIRContext *context, Location loc,
                   DictionaryAttr augmentedType, DictionaryAttr root,
                   StringAttr name, StringAttr defName,
                   std::optional<StringAttr> description, Twine path = {}) {
  auto classAttr =
      tryGetAs<StringAttr>(augmentedType, root, "class", loc, path);
  if (!classAttr)
    return std::nullopt;
  StringRef classBase = classAttr.getValue();
  if (!classBase.consume_front("sifive.enterprise.grandcentral.Augmented")) {
    mlir::emitError(loc,
                    "the 'class' was expected to start with "
                    "'sifive.enterprise.grandCentral.Augmented*', but was '" +
                        classAttr.getValue() + "' (Did you misspell it?)")
            .attachNote()
        << "see attribute: " << augmentedType;
    return std::nullopt;
  }

  // An AugmentedBundleType looks like:
  //   "defName": String
  //   "elements": Seq[AugmentedField]
  if (classBase == "BundleType") {
    defName = tryGetAs<StringAttr>(augmentedType, root, "defName", loc, path);
    if (!defName)
      return std::nullopt;

    // Each element is an AugmentedField with members:
    //   "name": String
    //   "description": Option[String]
    //   "tpe": AugmentedType
    SmallVector<Attribute> elements;
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, path);
    if (!elementsAttr)
      return std::nullopt;
    for (size_t i = 0, e = elementsAttr.size(); i != e; ++i) {
      auto field = dyn_cast_or_null<DictionaryAttr>(elementsAttr[i]);
      if (!field) {
        mlir::emitError(
            loc,
            "View 'info' attribute with path '.elements[" + Twine(i) +
                "]' contained an unexpected type (expected a DictionaryAttr).")
                .attachNote()
            << "The received element was: " << elementsAttr[i];
        return std::nullopt;
      }
      auto ePath = (path + ".elements[" + Twine(i) + "]").str();
      auto name = tryGetAs<StringAttr>(field, root, "name", loc, ePath);
      if (!name)
        return std::nullopt;
      auto tpe = tryGetAs<DictionaryAttr>(field, root, "tpe", loc, ePath);
      if (!tpe)
        return std::nullopt;
      std::optional<StringAttr> description;
      if (auto maybeDescription = field.get("description"))
        description = cast<StringAttr>(maybeDescription);
      auto eltAttr =
          parseAugmentedType(context, loc, tpe, root, name, defName,
                             description, path + "_" + name.getValue());
      if (!eltAttr)
        return std::nullopt;

      // Collect information necessary to build a module with this view later.
      // This includes the optional description and name.
      NamedAttrList attrs;
      if (auto maybeDescription = field.get("description"))
        attrs.append("description", cast<StringAttr>(maybeDescription));
      attrs.append("name", name);
      auto tpeClass = tpe.getAs<StringAttr>("class");
      if (!tpeClass) {
        mlir::emitError(loc, "missing 'class' key in") << tpe;
        return std::nullopt;
      }
      attrs.append("tpe", tpeClass);
      elements.push_back(*eltAttr);
    }
    // Add an attribute that stores information necessary to construct the
    // interface for the view.  This needs the name of the interface (defName)
    // and the names of the components inside it.
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    attrs.append("defName", defName);
    if (description)
      attrs.append("description", *description);
    attrs.append("elements", ArrayAttr::get(context, elements));
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // An AugmentedGroundType has no contents.
  if (classBase == "GroundType") {
    NamedAttrList elementIface;

    // Populate the attribute for the interface element.
    elementIface.append("class", classAttr);
    if (description)
      elementIface.append("description", *description);
    elementIface.append("name", name);

    return DictionaryAttr::getWithSorted(context, elementIface);
  }

  // An AugmentedVectorType looks like:
  //   "elements": Seq[AugmentedType]
  if (classBase == "VectorType") {
    auto elementsAttr =
        tryGetAs<ArrayAttr>(augmentedType, root, "elements", loc, path);
    if (!elementsAttr)
      return std::nullopt;
    SmallVector<Attribute> elements;
    for (auto [i, elt] : llvm::enumerate(elementsAttr)) {
      auto eltAttr = parseAugmentedType(
          context, loc, cast<DictionaryAttr>(elt), root, name,
          StringAttr::get(context, ""), std::nullopt, path + "_" + Twine(i));
      if (!eltAttr)
        return std::nullopt;
      elements.push_back(*eltAttr);
    }
    NamedAttrList attrs;
    attrs.append("class", classAttr);
    if (description)
      attrs.append("description", *description);
    attrs.append("elements", ArrayAttr::get(context, elements));
    attrs.append("name", name);
    return DictionaryAttr::getWithSorted(context, attrs);
  }

  // Anything else is unexpected or a user error if they manually wrote
  // the JSON/attribute.  Print an error and error out.
  mlir::emitError(loc, "found unknown AugmentedType '" + classAttr.getValue() +
                           "' (Did you misspell it?)")
          .attachNote()
      << "see attribute: " << augmentedType;
  return std::nullopt;
}

class ViewConverter : public IntrinsicConverter {
public:
  LogicalResult checkAndConvert(GenericIntrinsic gi,
                                GenericIntrinsicOpAdaptor adaptor,
                                PatternRewriter &rewriter) override {
    // Check structure of the intrinsic.
    if (gi.hasNoOutput() || gi.namedParam("info") || gi.namedParam("name") ||
        gi.namedParam("yaml", true))
      return failure();

    // Check operands.
    for (auto idx : llvm::seq(gi.getNumInputs()))
      if (gi.checkInputType(idx, "must be ground type", [](auto ty) {
            auto base = type_dyn_cast<FIRRTLBaseType>(ty);
            return base && base.isGround();
          }))
        return failure();

    // Parse "info" string parameter as JSON.
    auto view =
        llvm::json::parse(gi.getParamValue<StringAttr>("info").getValue());
    if (auto err = view.takeError()) {
      handleAllErrors(std::move(err), [&](const llvm::json::ParseError &a) {
        gi.emitError() << ": error parsing view JSON: " << a.message();
      });
      return failure();
    }

    // Convert JSON to MLIR attribute.
    llvm::json::Path::Root root;
    auto value = convertJSONToAttribute(gi.op.getContext(), view.get(), root);
    assert(value && "JSON to attribute failed but should not ever fail");

    // Check attribute is a dictionary, for AugmentedBundleTypeAttr
    // construction.
    auto dict = dyn_cast<DictionaryAttr>(value);
    if (!dict)
      return gi.emitError() << ": 'info' parameter must be a dictionary";

    auto nameAttr = gi.getParamValue<StringAttr>("name");
    auto result = parseAugmentedType(
        gi.op.getContext(), gi.op.getLoc(), dict, dict, nameAttr,
        /* defName= */ {}, /* description= */ std::nullopt);

    if (!result)
      return failure();

    // Build AugmentedBundleTypeAttr, unchecked.
    auto augmentedType =
        AugmentedBundleTypeAttr::get(gi.op.getContext(), *result);
    if (augmentedType.getClass() != augmentedBundleTypeAnnoClass)
      return gi.emitError() << ": 'info' must be augmented bundle";

    // Scan for ground-type (leaves) and count.
    SmallVector<DictionaryAttr> worklist;
    worklist.push_back(augmentedType.getUnderlying());
    size_t numLeaves = 0;
    auto augGroundAttr =
        StringAttr::get(gi.op.getContext(), augmentedGroundTypeAnnoClass);
    [[maybe_unused]] auto augBundleAttr =
        StringAttr::get(gi.op.getContext(), augmentedBundleTypeAnnoClass);
    [[maybe_unused]] auto augVectorAttr =
        StringAttr::get(gi.op.getContext(), augmentedVectorTypeAnnoClass);
    while (!worklist.empty()) {
      auto dict = worklist.pop_back_val();
      auto clazz = dict.getAs<StringAttr>("class");
      if (clazz == augGroundAttr) {
        ++numLeaves;
        continue;
      }
      assert(clazz == augBundleAttr || clazz == augVectorAttr);
      llvm::append_range(
          worklist,
          dict.getAs<ArrayAttr>("elements").getAsRange<DictionaryAttr>());
    }

    if (numLeaves != gi.getNumInputs())
      return gi.emitError()
             << " has " << gi.getNumInputs() << " operands but view 'info' has "
             << numLeaves << " leaf elements";

    // Check complete, convert!
    auto yaml = gi.getParamValue<StringAttr>("yaml");
    rewriter.replaceOpWithNewOp<ViewIntrinsicOp>(
        gi.op, nameAttr.getValue(), yaml, augmentedType, adaptor.getOperands());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// FIRRTL intrinsic lowering dialect interface
//===----------------------------------------------------------------------===//

#include "FIRRTLIntrinsics.cpp.inc"

void FIRRTLIntrinsicLoweringDialectInterface::populateIntrinsicLowerings(
    IntrinsicLowerings &lowering) const {
  populateLowerings(lowering);
  lowering.add<CirctDebugTypeDefConverter>("circt_debug_typedef");
  lowering.add<CirctDebugVarConverter>("circt_debug_var");
  lowering.add<CirctDebugModuleInfoConverter>("circt_debug_moduleinfo");
}

static constexpr StringLiteral kDebugLeavesAttr = "firrtl.debug_leaves";

namespace {

/// Process one `circt_debug_enumdef` intrinsic: validate, parse variants,
/// and (re)use a module-level `dbg.enumdef` op deduplicated by fqn.
LogicalResult processEnumDefIntrinsic(GenericIntrinsicOp op, FModuleOp mod,
                                      OpBuilder &builder,
                                      llvm::StringMap<Value> &seen) {
  auto *ctx = mod.getContext();
  GenericIntrinsic gi(op);
  auto fqn = gi.getParamValue<StringAttr>("fqn");
  auto typeName = gi.getParamValue<StringAttr>("typeName");
  auto variantsStr = gi.getParamValue<StringAttr>("variants");

  if (!fqn || !typeName || !variantsStr)
    return op.emitError("circt_debug_enumdef: missing required parameter(s) "
                        "'fqn', 'typeName', or 'variants'");

  auto parsed = llvm::json::parse(variantsStr.strref());
  if (auto err = parsed.takeError())
    return op.emitError("circt_debug_enumdef: failed to parse 'variants': ")
           << llvm::toString(std::move(err));
  auto *arr = parsed->getAsArray();
  if (!arr)
    return op.emitError("circt_debug_enumdef: 'variants' is not a JSON array");

  SmallVector<NamedAttribute> variants;
  for (const auto &item : *arr) {
    auto *obj = item.getAsObject();
    if (!obj)
      return op.emitError(
          "circt_debug_enumdef: variant entry is not a JSON object");
    auto varNameOpt = obj->getString("name");
    if (!varNameOpt)
      return op.emitError("circt_debug_enumdef: variant is missing 'name'");

    // Frontend emits value as a string; tolerate integers for hand-written
    // tests.
    int64_t val = 0;
    if (auto s = obj->getString("value")) {
      StringRef sv = *s;
      if (sv.getAsInteger(10, val))
        return op.emitError("circt_debug_enumdef: variant '")
               << *varNameOpt << "' has non-integer value '" << sv << "'";
    } else if (auto i = obj->getInteger("value")) {
      val = *i;
    } else {
      return op.emitError("circt_debug_enumdef: variant '")
             << *varNameOpt << "' is missing 'value'";
    }
    variants.push_back({StringAttr::get(ctx, *varNameOpt),
                        IntegerAttr::get(IntegerType::get(ctx, 64), val)});
  }

  auto variantsMap = DictionaryAttr::get(ctx, variants);

  // All emitted dbg.enumdef ops live at module scope (no scope operand on
  // the intrinsic). If per-scope enums ever land, extend the `seen` key to
  // include scope, in sync with EnumDefDeduplication.
  auto [it, inserted] = seen.try_emplace(fqn.getValue(), Value{});
  if (inserted) {
    builder.setInsertionPointToStart(mod.getBodyBlock());
    auto enumOp = debug::EnumDefOp::create(builder, op.getLoc(), typeName, fqn,
                                           variantsMap, /*scope=*/Value{});
    it->second = enumOp.getResult();
  } else if (auto existing = it->second.getDefiningOp<debug::EnumDefOp>()) {
    if (existing.getVariantsMap() != variantsMap)
      op.emitWarning("duplicate circt_debug_enumdef for fqn '")
          << fqn.getValue()
          << "' with differing variants; first definition wins";
  }
  return success();
}

/// Process one `circt_debug_subfield` intrinsic: validate parent/name
/// invariant and append a leaf entry for `firrtl.debug_leaves`.
LogicalResult processSubfieldIntrinsic(GenericIntrinsicOp op,
                                       SmallVectorImpl<Attribute> &entries) {
  auto *ctx = op.getContext();
  GenericIntrinsic gi(op);
  auto nameAttr = gi.getParamValue<StringAttr>("name");
  if (!nameAttr)
    return op.emitError(
        "circt_debug_subfield: missing required parameter 'name'");
  auto parentAttr = gi.getParamValue<StringAttr>("parent");
  if (!parentAttr)
    return op.emitError(
        "circt_debug_subfield: missing required parameter 'parent' "
        "(must equal the 'name' of the enclosing circt_debug_var)");

  // `name` must be a path rooted at `parent` — catches frontend regressions
  // that drop or mangle the root prefix.
  StringRef name = nameAttr.getValue();
  StringRef parent = parentAttr.getValue();
  if (!name.starts_with((parent + ".").str()) &&
      !name.starts_with((parent + "[").str()))
    return op.emitError("circt_debug_subfield: 'name' (")
           << name << ") is not rooted at 'parent' (" << parent
           << "); expected '" << parent << ".<field>' or '" << parent
           << "[<idx>]...'";

  SmallVector<NamedAttribute> fields;
  fields.push_back({StringAttr::get(ctx, "name"), nameAttr});
  fields.push_back({StringAttr::get(ctx, "parent"), parentAttr});
  if (auto tn = gi.getParamValue<StringAttr>("typeName"))
    fields.push_back({StringAttr::get(ctx, "typeName"), tn});
  if (auto fqn = gi.getParamValue<StringAttr>("enumFqn"))
    if (!fqn.getValue().empty())
      fields.push_back({StringAttr::get(ctx, "enumFqn"), fqn});
  if (auto paramsStr = gi.getParamValue<StringAttr>("params"))
    if (auto params = parseParamsJSON(ctx, paramsStr, op))
      fields.push_back({StringAttr::get(ctx, "params"), params});

  entries.push_back(DictionaryAttr::get(ctx, fields));
  return success();
}

} // namespace

namespace circt::firrtl {

LogicalResult liftDebugIntrinsics(FModuleOp mod, OpBuilder &builder) {
  auto *ctx = mod.getContext();
  llvm::StringMap<Value> seen;
  SmallVector<Attribute> leafEntries;
  SmallVector<GenericIntrinsicOp> toErase;
  bool hadError = false;

  mod.walk([&](GenericIntrinsicOp op) {
    auto kind = op.getIntrinsic();
    if (kind == "circt_debug_enumdef") {
      toErase.push_back(op);
      if (failed(processEnumDefIntrinsic(op, mod, builder, seen)))
        hadError = true;
    } else if (kind == "circt_debug_subfield") {
      toErase.push_back(op);
      if (failed(processSubfieldIntrinsic(op, leafEntries)))
        hadError = true;
    }
  });

  if (!leafEntries.empty())
    mod->setAttr(kDebugLeavesAttr, ArrayAttr::get(ctx, leafEntries));
  for (auto op : toErase)
    op.erase();
  return failure(hadError);
}

void clearDebugLeavesAttr(FModuleOp mod) { mod->removeAttr(kDebugLeavesAttr); }

} // namespace circt::firrtl
