//===- DebugOps.td - Debug dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
#define CIRCT_DIALECT_DEBUG_DEBUGOPS_TD

include "circt/Dialect/Debug/DebugDialect.td"
include "circt/Dialect/Debug/DebugTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class DebugOp<string mnemonic, list<Trait> traits = []> :
  Op<DebugDialect, mnemonic, traits>;


def ScopeOp : DebugOp<"scope"> {
  let summary = "Define a scope for debug values";
  let description = [{
    Creates an additional level of hierarchy in the DI, a "scope", which can be
    used to group variables and other scopes.

    Operations such as `hw.module` introduce an implicit scope. All debug
    operations within a module are added to that implicit scope, unless they
    have an explicit `scope` operand. Providing an explicit scope can be used to
    represent inlined modules.

    Scopes in DI do not necessarily have to correspond to levels of a module
    hierarchy. They can also be used to model things like control flow scopes,
    call stacks, and other source-language concepts.

    The `scope` operand of any debug dialect operation must be defined locally
    by a `dbg.scope` operation. It cannot be a block argument. (This is intended
    as a temporary restriction, to be lifted in the future.)
  }];
  let arguments = (ins
    StrAttr:$instanceName,
    StrAttr:$moduleName,
    Optional<ScopeType>:$scope
  );
  let results = (outs ScopeType:$result);
  let assemblyFormat = [{
    $instanceName `,` $moduleName (`scope` $scope^)? attr-dict
  }];
}


def VariableOp : DebugOp<"variable", [AttrSizedOperandSegments]> {
  let summary = "A named value to be captured in debug info";
  let description = [{
    Marks a value to be tracked in DI under the given name. The `dbg.variable`
    operation is useful to represent named values in a source language. For
    example, ports, constants, parameters, variables, nodes, or name aliases can
    all be represented as a variable. In combination with `dbg.array` and
    `dbg.struct`, complex aggregate source language values can be described and
    reconstituted from individual IR values. The `dbg.variable` operation acts
    as a tracker that follows the evolution of its assigned value throughout the
    compiler's pass pipelines. The debug info analysis uses this op to populate
    a module's scope with named source language values, and to establish how
    these source language values can be reconstituted from the actual IR values
    present at the end of compilation.

    See the rationale for examples and details. See the `dbg.scope` operation
    for additional details on how to use the `scope` operand.
  }];
  let arguments = (ins
    StrAttr:$name,
    AnyType:$value,
    OptionalAttr<StrAttr>:$typeName,
    OptionalAttr<ArrayAttr>:$params,
    Optional<EnumDefType>:$enumDef,
    Optional<ScopeType>:$scope
  );
  let assemblyFormat = [{
    $name `,` $value
    (`enumDef` $enumDef^)?
    attr-dict
    (`scope` $scope^)?
    `:` type($value)
  }];
}


def SubFieldOp : DebugOp<"subfield", [Pure]> {
  let summary = "A named subfield of an aggregate to be captured in debug info";
  let description = [{
    Like `dbg.variable` but represents a single named field within a Bundle
    or Vec aggregate. Returns a `!dbg.subfield` value consumed by `dbg.struct`
    or `dbg.array`. Has no `scope` operand — scope is inherited from the
    enclosing `dbg.variable`.
  }];
  let arguments = (ins
    StrAttr:$name,
    AnyType:$value,
    OptionalAttr<StrAttr>:$typeName,
    OptionalAttr<ArrayAttr>:$params,
    Optional<EnumDefType>:$enumDef
  );
  let results = (outs SubFieldType:$result);
  let hasCustomAssemblyFormat = 1;
}


def EnumDefOp : DebugOp<"enumdef", [Pure]> {
  let summary = "Define the named variants of a source-language enumeration";
  let description = [{
    Declares an enumeration by name and ordered list of variants. Each variant
    is a DictionaryAttr with keys `name` (StringAttr) and `value`
    (IntegerAttr i64). Emitted once per distinct enum type per module by the
    intrinsic lowering pass. Referenced by name via the `enumRef` attribute
    on `dbg.variable`.
  }];
  let arguments = (ins
    StrAttr:$name,
    StrAttr:$fqn,
    ArrayAttr:$variants
  );
  let results = (outs EnumDefType:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizer = true;
}


def ModuleInfoOp : DebugOp<"moduleinfo"> {
  let summary = "Source-language class name and location for the enclosing module";
  let description = [{
    Emitted once per hardware module by the intrinsic lowering pass. Carries
    the Chisel class name, instance name, source location and optional
    constructor parameters. Consumed by the hw-debug-info export pass.
  }];
  let arguments = (ins
    StrAttr:$typeName,
    StrAttr:$instanceName,
    StrAttr:$sourceFile,
    SI64Attr:$sourceLine,
    OptionalAttr<StrAttr>:$ctorParams
  );
  let assemblyFormat = [{
    $typeName `,` $instanceName `,` $sourceFile `:` $sourceLine
    (`params` $ctorParams^)?
    attr-dict
  }];
}


def StructOp : DebugOp<"struct", [
  Pure,
  PredOpTrait<"number of fields and names match",
    CPred<"$fields.size() == $names.size()">>
]> {
  let summary = "Aggregate values into a struct";
  let description = [{
    Creates a struct aggregate from a list of names and values. The `dbg.struct`
    operation allows for struct-like source language values to be captured in
    the debug info. This includes structs, unions, bidirectional bundles,
    interfaces, classes, and other similar structures.

    See the rationale for examples and details.
  }];
  let arguments = (ins Variadic<AnyType>:$fields, StrArrayAttr:$names);
  let results = (outs StructType:$result);
  let hasCustomAssemblyFormat = 1;
}


def ArrayOp : DebugOp<"array", [Pure, SameTypeOperands]> {
  let summary = "Aggregate values into an array";
  let description = [{
    Creates an array aggregate from a list of values. The first operand is
    placed at array index 0. The last operand is placed at the highest array
    index. The `dbg.array` operation allows for array-like source language
    values to be captured in the debug info. This includes arrays, or in the
    case of SystemVerilog, packed and unpacked arrays, lists, sequences, queues,
    FIFOs, channels, and vectors.

    See the rationale for examples and details.
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs ArrayType:$result);
  let hasCustomAssemblyFormat = 1;
}

def MemInfoOp : DebugOp<"meminfo"> {
  let summary = "Source-language memory metadata for the enclosing module";
  let description = [{
    Emitted once per memory declaration by the intrinsic lowering pass.
    Carries the memory name, kind (Mem vs SyncReadMem), depth, source
    location, data type descriptor, and optional read-under-write policy.
    Consumed by debug info export passes.
  }];
  let arguments = (ins
    StrAttr:$memName,
    StrAttr:$memoryKind,
    SI64Attr:$depth,
    StrAttr:$sourceFile,
    SI64Attr:$sourceLine,
    StrAttr:$dataType,
    OptionalAttr<StrAttr>:$readUnderWrite
  );
  let assemblyFormat = [{
    $memName `,` $memoryKind `,` $depth `,` $sourceFile `:` $sourceLine
    `,` `type` $dataType
    (`ruw` $readUnderWrite^)?
    attr-dict
  }];
}

#endif // CIRCT_DIALECT_DEBUG_DEBUGOPS_TD
