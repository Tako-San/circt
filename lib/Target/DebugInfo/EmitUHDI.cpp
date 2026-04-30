//===- EmitUHDI.cpp - Pool-based UHDI emission ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Serialise the compile-unit's debug information into the pool-based UHDI
// JSON format. Reads `dbg.*` ops stamped with `uhdi.stable_id`
// (firrtl-uhdi-init) and `uhdi.repr_entry` (hw-uhdi-verilog-snapshot);
// knows nothing else about the producer.
//
// Pools: representations (chisel/verilog), types (uint/sint/clock/struct/
// vector), expressions (comb opcode trees + aggregate '{ literals),
// variables (per-repr name/loc/value with sigName / exprRef / constant /
// bitVector binding), scopes (modules + inline + body[] from
// dbg.rootblock).
//
//===----------------------------------------------------------------------===//

#include "LocationUtils.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/VerilogName.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Target/DebugInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "emit-uhdi"

using namespace mlir;
using namespace circt;
using namespace debug;

using llvm::json::Array;
using llvm::json::Object;

namespace {

static constexpr StringRef kFormatVersion = "1.0";
static constexpr StringRef kChiselRepr = "chisel";

/// `dbg.subfield` is a transparent metadata wrapper (typeName / params /
/// enumDef) around a leaf SSA value, materialised by LowerIntrinsics on
/// circt_debug_subfield refs. For type interning, name resolution, and
/// expression-pool emission we want to see through it to the underlying
/// value (typically a flattened bundle port wire post-LowerTypes).
static mlir::Value unwrapDbgSubField(mlir::Value v) {
  while (auto opResult = dyn_cast<OpResult>(v))
    if (auto sf = dyn_cast<debug::SubFieldOp>(opResult.getOwner()))
      v = sf.getValue();
    else
      break;
  return v;
}

/// LowerTypes flattens an aggregate port into individual ports plus a
/// per-field wire that aliases the new flat port (so the original
/// `firrtl.subfield(io, idx)` uses can be RAUWed to a single SSA value).
/// After PrettifyVerilogNames the wire's `name` matches the port's name,
/// but its `hw.verilogName` is renamed (e.g. `io_a_0`) to dodge the
/// collision with the actual port. `sv::resolveVerilogName` returns the
/// renamed wire identifier, which is *not* the canonical signal a
/// VCD-bound wave consumer expects (and verilator's --trace may omit the
/// alias wire entirely). Walk through the wire-aliasing pattern to
/// recover the port name when one applies. Returns null if `value` is
/// not a recognisable port-alias.
static StringAttr resolvePortAliasName(mlir::Value value) {
  Operation *op = value.getDefiningOp();
  if (!op)
    return {};
  // Pre-ExportVerilog: hw.wire %port -> result. `getInput()` is the port.
  if (auto hwWire = dyn_cast<hw::WireOp>(op))
    if (auto ba = dyn_cast<BlockArgument>(hwWire.getInput()))
      if (auto mod =
              dyn_cast<hw::HWModuleOp>(ba.getOwner()->getParentOp()))
        return mod.getInputNameAttr(ba.getArgNumber());
  // Post-ExportVerilog: sv.read_inout %wire. Walk to the wire and
  // inspect its sv.assign / hw.output uses for port-alias patterns.
  Operation *wireOp = nullptr;
  if (auto rio = dyn_cast<sv::ReadInOutOp>(op)) {
    if (auto *def = rio.getInput().getDefiningOp())
      if (isa<sv::WireOp, sv::LogicOp>(def))
        wireOp = def;
  } else if (isa<sv::WireOp, sv::LogicOp>(op)) {
    wireOp = op;
  }
  if (!wireOp)
    return {};
  Value wireResult = wireOp->getResult(0);
  // Input pattern: `sv.assign %wire, %port_block_arg`.
  for (auto &use : wireResult.getUses()) {
    auto assign = dyn_cast<sv::AssignOp>(use.getOwner());
    if (!assign || use.getOperandNumber() != 0)
      continue;
    if (auto ba = dyn_cast<BlockArgument>(assign.getSrc()))
      if (auto mod =
              dyn_cast<hw::HWModuleOp>(ba.getOwner()->getParentOp()))
        return mod.getInputNameAttr(ba.getArgNumber());
  }
  // Output pattern: `%r = sv.read_inout %wire; hw.output %r, ...`.
  for (auto &use : wireResult.getUses())
    if (auto rio = dyn_cast<sv::ReadInOutOp>(use.getOwner()))
      for (auto &readUse : rio->getUses())
        if (auto out = dyn_cast<hw::OutputOp>(readUse.getOwner()))
          if (auto mod = out->getParentOfType<hw::HWModuleOp>())
            return mod.getOutputNameAttr(readUse.getOperandNumber());
  return {};
}

/// Combined leaf-name resolver: try the port-alias walk first (so bundle
/// fields surface as `io_a` rather than the lowering-introduced
/// `io_a_0`), then fall back to the standard SV-name resolver.
static StringAttr resolveBundleFieldName(mlir::Value value) {
  if (auto a = resolvePortAliasName(value))
    return a;
  return sv::resolveVerilogName(value);
}

//===----------------------------------------------------------------------===//
// File table & locations
//===----------------------------------------------------------------------===//

/// Per-repr file list, deduplicated, in insertion order.
struct FileTable {
  llvm::StringMap<unsigned> indexByPath;
  std::vector<std::string> ordered;

  unsigned intern(StringRef path) {
    auto it = indexByPath.find(path);
    if (it != indexByPath.end())
      return it->second;
    unsigned idx = ordered.size();
    ordered.emplace_back(path.str());
    indexByPath[path] = idx;
    return idx;
  }

  Array asArray() const {
    Array out;
    for (auto &s : ordered)
      out.push_back(s);
    return out;
  }
};

using circt::debuginfo::bestLocation;

/// Spec sec.6.4 Location object; nullopt if no FileLineColLoc.
static std::optional<Object>
locationObject(FileLineColLoc loc, StringRef prefix, FileTable &files) {
  if (!loc)
    return std::nullopt;
  SmallString<128> path;
  if (prefix.empty())
    path = loc.getFilename().getValue();
  else {
    path = prefix;
    llvm::sys::path::append(path, loc.getFilename().getValue());
  }
  Object o{{"file", int64_t(files.intern(path))}};
  if (loc.getLine() > 0)
    o["beginLine"] = int64_t(loc.getLine());
  if (loc.getColumn() > 0)
    o["beginColumn"] = int64_t(loc.getColumn());
  return o;
}

//===----------------------------------------------------------------------===//
// Type pool
//===----------------------------------------------------------------------===//

class TypePool {
public:
  /// Intern the type of `value`. Scalar ints map to canonical ground ids
  /// (`uint8`, `bool`, etc.); aggregates get a structural key + a
  /// hint-derived id (e.g. "BundleTest_io_in") or a `struct_N`/`array_N`
  /// fallback.
  std::string internValueType(mlir::Value value, StringRef nameHint = {}) {
    value = unwrapDbgSubField(value);
    if (auto opResult = dyn_cast<OpResult>(value)) {
      if (auto s = dyn_cast_or_null<debug::StructOp>(opResult.getOwner()))
        return internStruct(s, nameHint);
      if (auto a = dyn_cast_or_null<debug::ArrayOp>(opResult.getOwner()))
        return internArray(a, nameHint);
    }
    return internType(value.getType());
  }

  /// Intern an IR Type directly (synthesized-port fallback). Preserves
  /// !seq.clock as `kind: clock` (spec §4.2 GroundClock); opaque types
  /// fall back to `uint<0>` placeholder.
  std::string internType(mlir::Type type) {
    if (auto i = dyn_cast<IntegerType>(type))
      return internGround(i.isSigned() ? "sint" : "uint", i.getWidth());
    if (isa<seq::ClockType>(type))
      return internGround("clock", 0);
    return internGround("uint", 0);
  }

  Object asObject() const { return entries; }

private:
  Object entries;
  llvm::StringMap<std::string> byKey;
  unsigned structCounter = 0, arrayCounter = 0;

  std::string internGround(StringRef kind, unsigned width) {
    std::string id = (kind == "uint" && width == 1)
                         ? std::string("bool")
                         : (kind + std::to_string(width)).str();
    if (byKey.count(id))
      return id;
    byKey[id] = id;
    Object d{{"kind", kind.str()}};
    if (kind == "uint" || kind == "sint")
      d["width"] = int64_t(width);
    entries[id] = std::move(d);
    return id;
  }

  /// `flipped: true` when the SSA source is a BlockArgument (mirrors
  /// EmitHGLDD's direction-aware hgl_loc binding on bundle fields).
  std::string internStruct(debug::StructOp op, StringRef nameHint) {
    Array members;
    std::string key = "struct{";
    for (auto [nameAttr, field] : llvm::zip(op.getNames(), op.getFields())) {
      StringRef n = cast<StringAttr>(nameAttr).getValue();
      std::string childHint =
          nameHint.empty() ? std::string() : (nameHint + "_" + n).str();
      // dbg.subfield wraps the leaf-value with metadata (typeName, params)
      // — see through it for type interning AND for the BlockArgument check
      // (otherwise an input port flowing through a SubFieldOp loses its
      // `flipped` marker).
      mlir::Value innerField = unwrapDbgSubField(field);
      std::string ft = internValueType(innerField, childHint);
      bool flipped = isa<BlockArgument>(innerField);
      key.append(n).append(1, ':').append(ft);
      if (flipped)
        key += '!';
      key += ',';
      Object m{{"name", n.str()}, {"typeRef", ft}};
      if (flipped)
        m["flipped"] = true;
      members.push_back(std::move(m));
    }
    key += '}';
    if (auto it = byKey.find(key); it != byKey.end())
      return it->second;
    std::string id = pickAggregateId(nameHint, "struct_", structCounter);
    byKey[key] = id;
    entries[id] = Object{{"kind", "struct"}, {"members", std::move(members)}};
    return id;
  }

  std::string internArray(debug::ArrayOp op, StringRef nameHint) {
    auto elems = op.getElements();
    std::string elemId = elems.empty()
                             ? internGround("uint", 0)
                             : internValueType(elems.front(), nameHint);
    std::string key =
        "array{" + elemId + ':' + std::to_string(elems.size()) + '}';
    if (auto it = byKey.find(key); it != byKey.end())
      return it->second;
    std::string id = pickAggregateId(nameHint, "array_", arrayCounter);
    byKey[key] = id;
    entries[id] = Object{{"kind", "vector"},
                         {"elementRef", elemId},
                         {"size", int64_t(elems.size())}};
    return id;
  }

  std::string pickAggregateId(StringRef hint, StringRef prefix,
                              unsigned &counter) {
    if (hint.empty())
      return (prefix + std::to_string(counter++)).str();
    std::string id = hint.str();
    // Disambiguate on every prior occupant: `hint`, `hint_0`, `hint_1`, ...
    // are all checked, so a previously-issued `<hint>_0` (e.g. from a
    // different naming path) doesn't get re-issued.
    while (entries.get(id))
      id = (hint + "_" + std::to_string(counter++)).str();
    return id;
  }
};

//===----------------------------------------------------------------------===//
// Expression pool
//===----------------------------------------------------------------------===//

/// A `comb` op -> uhdi opcode (sec.5.2). Empty for variadic / mux /
/// concat / replicate / parity, which `operandFor` handles directly.
static StringRef combBinaryOpcode(Operation *op) {
  if (isa<comb::AndOp>(op))
    return "&";
  if (isa<comb::OrOp>(op))
    return "|";
  if (isa<comb::XorOp>(op))
    return "^";
  if (isa<comb::AddOp>(op))
    return "+";
  if (isa<comb::SubOp>(op))
    return "-";
  if (isa<comb::MulOp>(op))
    return "*";
  if (isa<comb::DivUOp>(op) || isa<comb::DivSOp>(op))
    return "/";
  if (isa<comb::ModUOp>(op) || isa<comb::ModSOp>(op))
    return "%";
  if (isa<comb::ShlOp>(op))
    return "<<";
  if (isa<comb::ShrUOp>(op))
    return ">>";
  if (isa<comb::ShrSOp>(op))
    return ">>>";
  if (auto cmp = dyn_cast<comb::ICmpOp>(op)) {
    using P = comb::ICmpPredicate;
    switch (cmp.getPredicate()) {
    case P::eq:
      return "==";
    case P::ne:
      return "!=";
    case P::ceq:
      return "===";
    case P::cne:
      return "!==";
    case P::weq:
      return "==?";
    case P::wne:
      return "!=?";
    case P::ult:
    case P::slt:
      return "<";
    case P::ugt:
    case P::sgt:
      return ">";
    case P::ule:
    case P::sle:
      return "<=";
    case P::uge:
    case P::sge:
      return ">=";
    }
  }
  return "";
}

/// Render a hw.constant: LeafConst for ≤63-bit values (cap at 63 so the
/// cast to int64_t never sign-flips), LeafBitVec otherwise. LeafBitVec
/// must not carry a width (schema additionalProperties:false).
static Object renderConstant(const llvm::APInt &val) {
  Object lit;
  unsigned bw = val.getBitWidth();
  if (val.getActiveBits() <= 63) {
    lit["constant"] = static_cast<int64_t>(val.getZExtValue());
    if (bw)
      lit["width"] = bw;
    return lit;
  }
  llvm::SmallString<128> bits;
  val.toString(bits, /*Radix=*/2, /*Signed=*/false,
               /*formatAsCLiteral=*/false, /*UpperCase=*/false,
               /*InsertSeparators=*/false);
  while (bits.size() < bw)
    bits.insert(bits.begin(), '0'); // toString drops leading zeros.
  lit["bitVector"] = bits.str().str();
  return lit;
}

class ExpressionPool {
public:
  /// uhdi Operand for `value`: prefers a sigName leaf when one resolves;
  /// aggregates / comb ops materialise a fresh pool entry and return
  /// `{exprRef: id}`. `leafNameFn` is the scalar-signal Verilog resolver.
  Object operandFor(mlir::Value value,
                    llvm::function_ref<StringAttr(mlir::Value)> leafNameFn) {
    // Look through dbg.subfield: a transparent metadata wrapper added by
    // LowerIntrinsics on circt_debug_subfield refs. Without this, the
    // bundle-field operands of a `dbg.struct` post-LowerTypes resolve to
    // empty sigNames (the SubFieldOp result has type !dbg.subfield, which
    // sv::resolveVerilogName doesn't know).
    value = unwrapDbgSubField(value);
    if (auto nameAttr = leafNameFn(value))
      return Object{{"sigName", nameAttr.getValue().str()}};
    auto opResult = dyn_cast<OpResult>(value);
    if (!opResult)
      return Object{{"sigName", ""}};
    Operation *defOp = opResult.getOwner();

    // Reserve the id BEFORE recursing so outer expressions get the
    // smaller index than the inner ones they depend on (matches HGLDD
    // ordering and avoids forward refs in the diff).
    auto wrap = [&](StringRef opcode, auto fillOperands) -> Object {
      std::string id = "expr_" + std::to_string(counter++);
      Array operands;
      fillOperands(operands);
      entries[id] =
          Object{{"opcode", opcode.str()}, {"operands", std::move(operands)}};
      return Object{{"exprRef", id}};
    };
    auto fillFrom = [&](auto range) {
      return [&, range](Array &out) {
        for (auto v : range)
          out.push_back(operandFor(v, leafNameFn));
      };
    };

    // dbg.expression result -> reference into the expressions pool by
    // its uhdi.stable_id (the entry itself is emitted in collect()).
    if (auto e = dyn_cast<debug::ExpressionOp>(defOp))
      if (auto id = e->getAttrOfType<StringAttr>(kUhdiStableIdAttr))
        return Object{{"exprRef", id.getValue().str()}};
    if (auto s = dyn_cast<debug::StructOp>(defOp))
      return wrap("'{", fillFrom(s.getFields()));
    if (auto a = dyn_cast<debug::ArrayOp>(defOp))
      return wrap("'{", fillFrom(a.getElements()));
    if (auto c = dyn_cast<hw::ConstantOp>(defOp))
      return renderConstant(c.getValue());
    if (auto concat = dyn_cast<comb::ConcatOp>(defOp))
      return wrap("{}", fillFrom(concat.getOperands()));
    if (auto repl = dyn_cast<comb::ReplicateOp>(defOp))
      return wrap("R{}", [&](Array &out) {
        out.push_back(operandFor(repl.getInput(), leafNameFn));
        out.push_back(Object{{"constant", int64_t(repl.getMultiple())}});
      });
    if (auto mux = dyn_cast<comb::MuxOp>(defOp))
      return wrap("?:", [&](Array &out) {
        out.push_back(operandFor(mux.getCond(), leafNameFn));
        out.push_back(operandFor(mux.getTrueValue(), leafNameFn));
        out.push_back(operandFor(mux.getFalseValue(), leafNameFn));
      });
    if (isa<comb::ParityOp>(defOp) && defOp->getNumOperands() == 1)
      return wrap("^", fillFrom(defOp->getOperands()));
    if (StringRef opcode = combBinaryOpcode(defOp);
        !opcode.empty() && defOp->getNumOperands() == 2)
      return wrap(opcode, fillFrom(defOp->getOperands()));

    // Last resort: caller (emitVariable) detects this trivial-empty
    // sigName and suppresses the value rather than emitting `{sigName: ""}`.
    return Object{{"sigName", ""}};
  }

  Object asObject() const { return entries; }

  /// Insert an externally-keyed entry (used for `dbg.expression` ops
  /// whose key is the uhdi.stable_id rather than an auto-generated
  /// `expr_N`). No-op if the key is already populated.
  void insertEntry(StringRef key, Object entry) {
    if (!entries.get(key))
      entries[key] = std::move(entry);
  }

private:
  Object entries;
  unsigned counter = 0;
};

//===----------------------------------------------------------------------===//
// Verilog-name resolver
//===----------------------------------------------------------------------===//

/// Read the per-repr `name` from an op's uhdi.repr_entry stamp, if any.
static StringAttr readReprName(Operation *op, StringRef reprKey) {
  auto dict = op->getAttrOfType<DictionaryAttr>(kUhdiReprEntryAttr);
  if (!dict)
    return {};
  auto inner = dict.getNamed(reprKey);
  if (!inner)
    return {};
  if (auto innerDict = dyn_cast<DictionaryAttr>(inner->getValue()))
    return innerDict.getAs<StringAttr>("name");
  return {};
}

/// Live IR walk wins over the snapshot stamp (the walk sees ExportVerilog's
/// freshly spilled wires; a stale stamp would shadow them).
static StringAttr bestVerilogName(Operation *op, mlir::Value value) {
  if (auto name = sv::resolveVerilogName(value))
    return name;
  return readReprName(op, kUhdiVerilogRepr);
}

//===----------------------------------------------------------------------===//
// Variable / scope assembly
//===----------------------------------------------------------------------===//

/// Owner scope for a dbg op: the enclosing `dbg.scope`'s stable_id when
/// present, else the enclosing hw.module's stable_id (or symbol name).
static StringRef ownerScopeId(Operation *op) {
  if (auto var = dyn_cast<debug::VariableOp>(op))
    if (auto scope = var.getScope())
      if (auto *def = scope.getDefiningOp())
        if (auto id = def->getAttrOfType<StringAttr>(kUhdiStableIdAttr))
          return id.getValue();
  for (auto *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (auto id = p->getAttrOfType<StringAttr>(kUhdiStableIdAttr))
      return id.getValue();
    if (isa<hw::HWModuleOp, hw::HWModuleExternOp>(p))
      break;
  }
  if (auto mod = op->getParentOfType<hw::HWModuleOp>())
    return mod.getNameAttr().getValue();
  if (auto mod = op->getParentOfType<hw::HWModuleExternOp>())
    return mod.getNameAttr().getValue();
  return {};
}

namespace {
/// Bundle of per-document mutable state passed through the assembly
/// helpers. Keeps the function signatures from sprouting a half-dozen
/// args each.
struct EmitState {
  TypePool &types;
  ExpressionPool &exprs;
  FileTable &chiselFiles, &verilogFiles;
  StringRef chiselPrefix, verilogPrefix;
  bool onlyExisting;
  llvm::StringMap<bool> &existsCache;

  std::optional<Object> chiselLoc(Location loc) {
    return locationObject(
        bestLocation(loc, /*emitted=*/false, onlyExisting, &existsCache),
        chiselPrefix, chiselFiles);
  }
  std::optional<Object> verilogLoc(Location loc) {
    return locationObject(
        bestLocation(loc, /*emitted=*/true, onlyExisting, &existsCache),
        verilogPrefix, verilogFiles);
  }
};
} // namespace

/// One `variables[id]` entry for a `dbg.variable`.
static Object emitVariable(debug::VariableOp var, EmitState &s) {
  Object entry;
  // Hint nested struct/array ids with `<Module>_<var>[_field]...` to
  // match HGLDD's naming so the canonical diff stays name-equal.
  std::string structHint;
  if (isa_and_nonnull<debug::StructOp, debug::ArrayOp>(
          var.getValue().getDefiningOp())) {
    if (auto mod = var->getParentOfType<hw::HWModuleOp>())
      structHint = (mod.getNameAttr().getValue() + "_" + var.getName()).str();
    else
      structHint = var.getName().str();
  }
  entry["typeRef"] = s.types.internValueType(var.getValue(), structHint);
  if (StringRef owner = ownerScopeId(var); !owner.empty())
    entry["ownerScopeRef"] = owner.str();

  // BlockArgument of an hw.module is always an input port; everything else
  // is a node (named wire / reg / intermediate).
  bool isPort = false;
  if (auto blockArg = dyn_cast<BlockArgument>(var.getValue()))
    isPort = isa<hw::HWModuleOp>(blockArg.getOwner()->getParentOp());
  entry["bindKind"] = isPort ? "port" : "node";
  if (isPort)
    entry["direction"] = "input";

  Object reprs;
  Object chisel{{"name", var.getName().str()}};
  if (auto loc = s.chiselLoc(var.getLoc()))
    chisel["location"] = std::move(*loc);
  // Spec linter wants `status` set explicitly; omitted reads as "emitter
  // didn't check" rather than "preserved".
  chisel["status"] = "preserved";
  reprs[kChiselRepr] = std::move(chisel);

  // Verilog value: aggregate -> exprRef pool; constant -> constant /
  // bitVector; otherwise -> sigName via bestVerilogName with operandFor
  // as the DCE'd-/inlined fallback.
  Object verilog;
  bool haveValue = false;
  // Bundle fields (the only place dbg.subfield surfaces) end up as
  // wires aliasing flat module ports post-LowerTypes: prefer the port
  // name over the lowering-introduced wire identifier.
  auto leafName = [](mlir::Value v) { return resolveBundleFieldName(v); };
  bool isAggregate = isa_and_nonnull<debug::StructOp, debug::ArrayOp>(
      var.getValue().getDefiningOp());

  if (isAggregate) {
    verilog["value"] = s.exprs.operandFor(var.getValue(), leafName);
    haveValue = true;
  } else if (auto opResult = dyn_cast<OpResult>(var.getValue());
             opResult && isa_and_nonnull<hw::ConstantOp>(opResult.getOwner())) {
    Object lit =
        renderConstant(cast<hw::ConstantOp>(opResult.getOwner()).getValue());
    // Per variables.schema.json ValueBinding: drop the auxiliary `width`
    // (LeafConst variant only has constant, LeafBitVec only bitVector).
    lit.erase("width");
    verilog["value"] = std::move(lit);
    haveValue = true;
  } else if (auto vname = bestVerilogName(var, var.getValue())) {
    verilog["name"] = vname.getValue().str();
    verilog["value"] = Object{{"sigName", vname.getValue().str()}};
    haveValue = true;
  } else if (dyn_cast<OpResult>(var.getValue())) {
    Object expr = s.exprs.operandFor(var.getValue(), leafName);
    bool trivialEmpty = (expr.size() == 1) &&
                        (expr.find("sigName") != expr.end()) &&
                        (expr["sigName"] == "");
    if (!trivialEmpty) {
      verilog["value"] = std::move(expr);
      haveValue = true;
    }
  }
  if (haveValue) {
    if (auto loc = s.verilogLoc(var.getLoc()))
      verilog["location"] = std::move(*loc);
    verilog["status"] = "preserved";
    reprs[kUhdiVerilogRepr] = std::move(verilog);
  }
  entry["representations"] = std::move(reprs);
  return entry;
}

/// Build a `{name?, location?}` per-repr dict (used by inline scopes,
/// instantiates, module reprs).
static Object reprDict(StringRef name, std::optional<Object> loc) {
  Object d;
  if (!name.empty())
    d["name"] = name.str();
  if (loc)
    d["location"] = std::move(*loc);
  return d;
}

/// One `scopes[id]` entry for an `hw.module`.
static Object emitModuleScope(hw::HWModuleOp module, EmitState &s) {
  StringRef sym = module.getNameAttr().getValue();
  StringRef vname = sym;
  if (auto attr = module->getAttrOfType<StringAttr>("verilogName"))
    vname = attr.getValue();
  Object entry{{"name", sym.str()}, {"kind", "module"}};
  Object reprs;
  reprs[kChiselRepr] = reprDict(sym, s.chiselLoc(module.getLoc()));
  reprs[kUhdiVerilogRepr] = reprDict(vname, s.verilogLoc(module.getLoc()));
  entry["representations"] = std::move(reprs);

  // hw.instance children -> instantiates[].
  Array instantiates;
  module.walk([&](hw::InstanceOp inst) {
    Object e{{"as", inst.getInstanceName().str()},
             {"scopeRef", inst.getModuleName().str()}};
    Object perReprs;
    Object chisel = reprDict("", s.chiselLoc(inst.getLoc()));
    if (!chisel.empty())
      perReprs[kChiselRepr] = std::move(chisel);
    // PrettifyVerilogNames may rename the instance; record only when it
    // diverges from the source-level instanceName.
    StringRef vrename;
    if (auto vn = inst->getAttrOfType<StringAttr>("hw.verilogName");
        vn && vn.getValue() != inst.getInstanceName())
      vrename = vn.getValue();
    Object verilog = reprDict(vrename, s.verilogLoc(inst.getLoc()));
    if (!verilog.empty())
      perReprs[kUhdiVerilogRepr] = std::move(verilog);
    if (!perReprs.empty())
      e["representations"] = std::move(perReprs);
    instantiates.push_back(std::move(e));
  });
  if (!instantiates.empty())
    entry["instantiates"] = std::move(instantiates);
  return entry;
}

/// One `scopes[id]` entry for an inline `dbg.scope`.
/// `containerScopeRef` lets the HGLDD converter graft the inline scope
/// into the right parent's `children[]` array.
static Object emitInlineScope(debug::ScopeOp scope, EmitState &s) {
  Object entry{{"name", scope.getModuleName().str()}, {"kind", "inline"}};
  Object reprs;
  reprs[kChiselRepr] =
      reprDict(scope.getInstanceName(), s.chiselLoc(scope.getLoc()));
  if (auto vname = readReprName(scope, kUhdiVerilogRepr))
    reprs[kUhdiVerilogRepr] = Object{{"name", vname.getValue().str()}};
  entry["representations"] = std::move(reprs);
  if (auto module = scope->getParentOfType<hw::HWModuleOp>())
    entry["containerScopeRef"] = module.getNameAttr().getValue().str();
  return entry;
}

//===----------------------------------------------------------------------===//
// Scope body
//===----------------------------------------------------------------------===//

/// `name -> stable_id` (or sig_name) map for statement-tree refs. Entries:
/// dbg.variable / dbg.expression names -> their stable_id; struct members
/// -> the field's Verilog sig_name. Collisions poison the entry to an
/// empty string, and `resolveVarRef` falls back to the literal name.
struct VarRefIndex {
  llvm::StringMap<std::string> map; // empty value = poisoned/ambiguous.
};

/// Index each `dbg.struct` member as `<parent>.<field>` -> its sig_name
/// (live HW IR). Recurses for nested structs.
static void indexStructFields(VarRefIndex &idx, StringRef parentName,
                              debug::StructOp structOp) {
  for (auto [nameAttr, fieldVal] :
       llvm::zip(structOp.getNames(), structOp.getFields())) {
    StringRef fieldName = cast<StringAttr>(nameAttr).getValue();
    std::string path = (parentName + "." + fieldName).str();
    // dbg.subfield wraps the leaf with metadata; unwrap so resolveVerilogName
    // sees the underlying wire/port and the nested-struct check below picks
    // up `dbg.subfield(dbg.struct ...)` shapes too.
    mlir::Value innerField = unwrapDbgSubField(fieldVal);
    if (auto vname = resolveBundleFieldName(innerField)) {
      idx.map.try_emplace(path, vname.getValue().str());
    }
    if (auto *fieldDefOp = innerField.getDefiningOp())
      if (auto nested = dyn_cast<debug::StructOp>(fieldDefOp))
        indexStructFields(idx, path, nested);
  }
}

/// Add `var` (and any flat struct-member subfields it covers) to its
/// owner module's VarRefIndex. Collisions poison the entry to "" so
/// `resolveVarRef` falls back to the literal name.
static void addVarToIndex(VarRefIndex &idx, debug::VariableOp var,
                          hw::HWModuleOp mod) {
  auto id = var->getAttrOfType<StringAttr>(kUhdiStableIdAttr);
  if (!id)
    return;
  auto [it, inserted] = idx.map.try_emplace(var.getName(), id.getValue().str());
  if (!inserted && it->second != id.getValue()) {
    var.emitWarning() << "uhdi: dbg.variable name '" << var.getName()
                      << "' collides within module '" << mod.getName()
                      << "'; body varRef tokens for this name will be "
                         "left unresolved";
    it->second = ""; // ambiguous sentinel
  }
  if (auto *defOp = var.getValue().getDefiningOp())
    if (auto structOp = dyn_cast<debug::StructOp>(defOp))
      indexStructFields(idx, var.getName(), structOp);
}

/// dbg.expression is a sibling value-tracker — same index so `guardRef` /
/// `enableRef` tokens resolve uniformly regardless of whether they name a
/// variable or a materialised compound expression.
static void addExprToIndex(VarRefIndex &idx, debug::ExpressionOp expr,
                           hw::HWModuleOp mod) {
  auto id = expr->getAttrOfType<StringAttr>(kUhdiStableIdAttr);
  if (!id)
    return;
  auto [it, inserted] =
      idx.map.try_emplace(expr.getName(), id.getValue().str());
  if (!inserted && it->second != id.getValue()) {
    expr.emitWarning() << "uhdi: dbg.expression name '" << expr.getName()
                       << "' collides in module '" << mod.getName()
                       << "'; refs will be left unresolved";
    it->second = "";
  }
}

static std::string resolveVarRef(const VarRefIndex &index, StringRef name) {
  auto it = index.map.find(name);
  if (it == index.map.end() || it->second.empty())
    return name.str();
  return it->second;
}

/// enableRef shape: `&`-joined stable_ids with optional `!` per leaf.
static Object emitBreakpointMeta(DictionaryAttr bp,
                                 const VarRefIndex &varIndex) {
  if (!bp)
    return {};
  auto enable = bp.getAs<StringAttr>("enableRef");
  if (!enable)
    return {};
  std::string resolved;
  SmallVector<StringRef, 4> tokens;
  enable.getValue().split(tokens, '&', -1, /*KeepEmpty=*/false);
  for (auto tok : tokens) {
    bool negated = tok.starts_with("!");
    if (negated)
      tok = tok.drop_front();
    if (!resolved.empty())
      resolved += '&';
    if (negated)
      resolved += '!';
    resolved += resolveVarRef(varIndex, tok);
  }
  return Object{{"enableRef", resolved}};
}

/// Per-statement `locations` map (chisel/verilog).
static void attachLocations(Object &entry, Operation *op, EmitState &s) {
  Object locs;
  if (auto loc = s.chiselLoc(op->getLoc()))
    locs[kChiselRepr] = std::move(*loc);
  if (auto loc = s.verilogLoc(op->getLoc()))
    locs[kUhdiVerilogRepr] = std::move(*loc);
  if (!locs.empty())
    entry["locations"] = std::move(locs);
}

static Array emitStatementList(Region &region, const VarRefIndex &varIndex,
                               EmitState &s) {
  Array body;
  if (region.empty())
    return body;
  for (Operation &op : region.front()) {
    Object entry;
    if (auto connect = dyn_cast<debug::ConnectStmtOp>(op)) {
      entry["kind"] = "connect";
      entry["varRef"] = resolveVarRef(varIndex, connect.getVarRef());
      entry["valueRef"] =
          Object{{"varRef", resolveVarRef(varIndex, connect.getValueRef())}};
      if (auto bp = emitBreakpointMeta(connect.getBpAttr(), varIndex);
          !bp.empty())
        entry["bp"] = std::move(bp);
    } else if (auto block = dyn_cast<debug::SubBlockOp>(op)) {
      entry["kind"] = "block";
      entry["guardRef"] = resolveVarRef(varIndex, block.getGuardRef());
      if (block.getNegated())
        entry["negated"] = true;
      entry["body"] = emitStatementList(block.getBody(), varIndex, s);
    } else if (auto decl = dyn_cast<debug::DeclStmtOp>(op)) {
      entry["kind"] = "decl";
      entry["varRef"] = resolveVarRef(varIndex, decl.getVarRef());
    } else {
      continue;
    }
    attachLocations(entry, &op, s);
    body.push_back(std::move(entry));
  }
  return body;
}

//===----------------------------------------------------------------------===//
// Emitter driver
//===----------------------------------------------------------------------===//

class UhdiEmitter {
public:
  UhdiEmitter(Operation *root, const EmitUHDIOptions &options)
      : root(root), options(options) {}
  LogicalResult run(raw_ostream &os);

private:
  Operation *root;
  const EmitUHDIOptions &options;
  TypePool types;
  ExpressionPool exprs;
  FileTable chiselFiles, verilogFiles;
  Object variables, scopes;
  SmallVector<std::string> topScopes;
  llvm::StringMap<bool> existsCache;

  void collect(mlir::ModuleOp top);
  Object render() const;

  EmitState state() {
    return EmitState{types,
                     exprs,
                     chiselFiles,
                     verilogFiles,
                     options.sourceFilePrefix,
                     options.outputFilePrefix,
                     options.onlyExistingFileLocs,
                     existsCache};
  }

  /// Fill in port_var entries for ports that lack a `dbg.variable` cover
  /// (e.g. firtool-emitted SRAM macros, or hand-written hw.module fixtures
  /// with no MaterializeDebugInfo run). Per-port — modules that already
  /// have dbg.variables for *some* of their ports still get synthesized
  /// entries for the *uncovered* ports, so partial coverage doesn't read
  /// as full coverage. Output ports omit the verilog repr (matches native
  /// HGLDD: the macro generates the backing signal outside the dbg graph).
  void synthesizePortVars(hw::HWModuleOp mod,
                          llvm::StringMap<Array> &orderedVarsByScope,
                          const llvm::DenseSet<unsigned> &coveredPortIndices);
};

void UhdiEmitter::synthesizePortVars(
    hw::HWModuleOp mod, llvm::StringMap<Array> &orderedVarsByScope,
    const llvm::DenseSet<unsigned> &coveredPortIndices) {
  StringRef scopeKey = mod.getNameAttr().getValue();
  auto modType = mod.getModuleType();
  if (modType.getNumPorts() == 0)
    return;

  Array &refs = orderedVarsByScope[scopeKey];
  for (size_t i = 0, e = modType.getNumPorts(); i < e; ++i) {
    if (coveredPortIndices.contains(i))
      continue;
    StringRef portName = modType.getPortName(i);
    if (portName.empty())
      continue;
    bool isOutput = modType.isOutput(i);
    std::string varId = (Twine("var_") + scopeKey + "_" + portName).str();
    Object entry{{"ownerScopeRef", scopeKey.str()},
                 {"bindKind", "port"},
                 {"direction", isOutput ? "output" : "input"},
                 {"typeRef", types.internType(modType.getPorts()[i].type)}};
    Object reprs;
    reprs[kChiselRepr] = Object{{"name", portName.str()}};
    if (!isOutput)
      reprs[kUhdiVerilogRepr] =
          Object{{"name", portName.str()},
                 {"value", Object{{"sigName", portName.str()}}}};
    entry["representations"] = std::move(reprs);
    variables[varId] = std::move(entry);
    refs.push_back(varId);
  }
}

void UhdiEmitter::collect(mlir::ModuleOp top) {
  EmitState s = state();

  // 1. Module / extmodule scopes; record public modules as `top`.
  // (uhdi-init doesn't stamp modules; the symbol name stands in.)
  for (auto &op : top.getOps()) {
    if (auto mod = dyn_cast<hw::HWModuleOp>(op)) {
      scopes[mod.getNameAttr().getValue().str()] = emitModuleScope(mod, s);
      if (mod.isPublic())
        topScopes.push_back(mod.getNameAttr().getValue().str());
    } else if (auto ext = dyn_cast<hw::HWModuleExternOp>(op)) {
      scopes[ext.getNameAttr().getValue().str()] = Object{
          {"name", ext.getNameAttr().getValue().str()}, {"kind", "extmodule"}};
    }
  }

  // 2. One whole-circuit walk that does step 2's variable/scope/expression
  // emission AND collects per-module indexes consumed by steps 3 and 4
  // (port coverage, varRef map, root block) — so steps 3/4 don't re-walk.
  llvm::StringMap<Array> orderedVarsByScope;
  llvm::DenseMap<hw::HWModuleOp, llvm::DenseSet<unsigned>> coveredPortsByMod;
  llvm::DenseMap<hw::HWModuleOp, VarRefIndex> varRefIndexByMod;
  llvm::DenseMap<hw::HWModuleOp, debug::RootBlockOp> rootBlockByMod;
  top.walk([&](Operation *inner) {
    if (auto scope = dyn_cast<debug::ScopeOp>(inner)) {
      if (auto id = scope->getAttrOfType<StringAttr>(kUhdiStableIdAttr))
        scopes[id.getValue().str()] = emitInlineScope(scope, s);
      return;
    }
    if (auto var = dyn_cast<debug::VariableOp>(inner)) {
      auto id = var->getAttrOfType<StringAttr>(kUhdiStableIdAttr);
      if (!id)
        return;
      variables[id.getValue().str()] = emitVariable(var, s);
      if (StringRef owner = ownerScopeId(var); !owner.empty())
        orderedVarsByScope[owner].push_back(id.getValue().str());
      if (auto mod = var->getParentOfType<hw::HWModuleOp>()) {
        addVarToIndex(varRefIndexByMod[mod], var, mod);
        if (auto blockArg = dyn_cast<BlockArgument>(var.getValue()))
          if (mod.getBodyRegion().hasOneBlock() &&
              blockArg.getOwner() == &mod.getBodyRegion().front())
            coveredPortsByMod[mod].insert(blockArg.getArgNumber());
      }
      return;
    }
    if (auto expr = dyn_cast<debug::ExpressionOp>(inner)) {
      // dbg.expression: compound expression captured by capture-when for
      // `when` guards / non-trivial valueRefs. Serialise into the
      // expressions pool keyed by stable_id so statement-tree refs
      // resolve into a real spec §5 opcode tree.
      auto id = expr->getAttrOfType<StringAttr>(kUhdiStableIdAttr);
      if (!id)
        return;
      Array operands;
      auto leafName = [](mlir::Value v) { return resolveBundleFieldName(v); };
      for (auto operand : expr.getExprOperands())
        operands.push_back(s.exprs.operandFor(operand, leafName));
      s.exprs.insertEntry(id.getValue(),
                          Object{{"opcode", expr.getOpcode().str()},
                                 {"operands", std::move(operands)}});
      if (auto mod = expr->getParentOfType<hw::HWModuleOp>())
        addExprToIndex(varRefIndexByMod[mod], expr, mod);
      return;
    }
    if (auto rb = dyn_cast<debug::RootBlockOp>(inner))
      if (auto mod = rb->getParentOfType<hw::HWModuleOp>())
        rootBlockByMod.try_emplace(mod, rb);
  });

  // 3. Synthesized port_vars for modules with no dbg.variable coverage.
  for (auto &op : top.getOps())
    if (auto mod = dyn_cast<hw::HWModuleOp>(op))
      synthesizePortVars(mod, orderedVarsByScope, coveredPortsByMod[mod]);

  // Stitch ordered variableRefs onto each scope entry.
  for (auto &kv : orderedVarsByScope) {
    auto it = scopes.find(kv.first());
    if (it == scopes.end())
      continue;
    if (auto *obj = it->getSecond().getAsObject())
      (*obj)["variableRefs"] = std::move(kv.second);
  }

  // 4. Per-module body[] from dbg.rootblock (placed by capture-when).
  // Inlined-scope bodies are not captured.
  for (auto &op : top.getOps()) {
    auto mod = dyn_cast<hw::HWModuleOp>(op);
    if (!mod)
      continue;
    auto rbIt = rootBlockByMod.find(mod);
    if (rbIt == rootBlockByMod.end())
      continue;
    auto it = scopes.find(mod.getNameAttr().getValue());
    if (it == scopes.end())
      continue;
    if (auto *obj = it->getSecond().getAsObject()) {
      Array body =
          emitStatementList(rbIt->second.getBody(), varRefIndexByMod[mod], s);
      if (!body.empty())
        (*obj)["body"] = std::move(body);
    }
  }

  // 5. Synthetic `_uhdi_empty_design` placeholder so an extmodule-only /
  // empty input still satisfies the schema's `top: minItems=1` invariant.
  bool haveModule = false;
  for (auto &kv : scopes)
    if (auto *obj = kv.getSecond().getAsObject())
      if (auto k = obj->getString("kind"); k && *k == "module") {
        haveModule = true;
        break;
      }
  if (!haveModule) {
    top.emitWarning() << "uhdi: no hw.module-kind scope found; emitting "
                         "synthetic '_uhdi_empty_design' top scope so the "
                         "document remains schema-valid";
    static constexpr StringRef kEmpty = "_uhdi_empty_design";
    scopes[kEmpty] = Object{{"name", kEmpty.str()}, {"kind", "module"}};
  }
}

Object UhdiEmitter::render() const {
  Object doc;
  doc["format"] = Object{{"name", "uhdi"}, {"version", kFormatVersion.str()}};
  doc["producer"] = Object{{"name", "circt"}};
  doc["representations"] =
      Object{{kChiselRepr, Object{{"kind", "source"},
                                  {"language", "Chisel"},
                                  {"files", chiselFiles.asArray()}}},
             {kUhdiVerilogRepr, Object{{"kind", "hdl"},
                                       {"language", "SystemVerilog"},
                                       {"files", verilogFiles.asArray()}}}};
  doc["roles"] = Object{{"authoring", kChiselRepr.str()},
                        {"simulation", kUhdiVerilogRepr.str()},
                        {"canonical", kUhdiVerilogRepr.str()}};

  Array topArray;
  for (auto &s : topScopes)
    topArray.push_back(s);
  // No public hw.module: prefer first `module`-kind scope (an extmodule
  // chosen as top would mislead consumers). The `_uhdi_empty_design`
  // placeholder is materialised in collect() so this stays const.
  if (topArray.empty())
    for (auto &kv : scopes)
      if (auto *obj = kv.getSecond().getAsObject())
        if (auto k = obj->getString("kind"); k && *k == "module") {
          topArray.push_back(kv.getFirst().str());
          break;
        }
  doc["top"] = std::move(topArray);

  doc["types"] = types.asObject();
  if (Object e = exprs.asObject(); !e.empty())
    doc["expressions"] = std::move(e);
  doc["variables"] = Object(variables);
  doc["scopes"] = Object(scopes);
  return doc;
}

LogicalResult UhdiEmitter::run(raw_ostream &os) {
  auto top = dyn_cast<mlir::ModuleOp>(root);
  if (!top)
    return root->emitError("EmitUHDI expects a top-level builtin.module");
  collect(top);
  llvm::json::Value rendered(render());
  llvm::json::OStream(os, /*IndentSize=*/2).value(rendered);
  os << "\n";
  return success();
}

} // namespace

LogicalResult debug::emitUHDI(Operation *module, raw_ostream &os,
                              const EmitUHDIOptions &options) {
  return UhdiEmitter(module, options).run(os);
}
