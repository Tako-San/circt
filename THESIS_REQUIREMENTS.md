# Technical Requirements: Debug Intrinsics Lowering for Chisel

**Document Version:** 1.2  
**Date:** 2026-01-18  
**Author:** Tako-San (Thesis Project)  
**Status:** 🟢 Building Phase  

---

## 🧪 Acceptance Testing Methodology

### Philosophy

Каждое требование (REQ-X.Y) имеет acceptance criteria (AC-X.Y.Z). Для **каждого AC** должна быть:
1. **Конкретная команда** для проверки
2. **Ожидаемый результат** (pass/fail, конкретный output)
3. **Способ воспроизведения** (если fail → как отладить)

Это гарантирует, что:
- Вы точно знаете, когда задача выполнена
- Можно легко показать прогресс в дипломе
- Регрессии обнаруживаются немедленно

### Phase 1: Verification Checklist

#### AC-1.1.1: EnumDefOp compiles without errors

**Команда:**
```bash
cd build
ninja bin/circt-opt 2>&1 | grep -i "enumdefop\|error"
```

**Ожидаемый результат:**
- Exit code: 0
- Никаких строк с "error" в выводе
- Файл `bin/circt-opt` обновлён (проверить timestamp)

**Если fail:**
```bash
# Смотрим полный лог компиляции
ninja -v bin/circt-opt > build.log 2>&1
grep -A5 -B5 "EnumDefOp" build.log
```

---

#### AC-1.1.2: ModuleInfoOp compiles without errors

**Команда:** Аналогично AC-1.1.1, заменить `EnumDefOp` на `ModuleInfoOp`.

---

#### AC-1.1.3: SubfieldOp compiles without errors

**Команда:** Аналогично AC-1.1.1, заменить `EnumDefOp` на `SubfieldOp`.

---

#### AC-1.1.4: `ninja bin/circt-opt` succeeds

**Команда:**
```bash
cd build
time ninja bin/circt-opt
echo "Exit code: $?"
./bin/circt-opt --version
```

**Ожидаемый результат:**
```
[100/100] Linking CXX executable bin/circt-opt
Exit code: 0
LLVM version 22.0.0git
CIRCT 2c385b2c0
```

---

#### AC-1.1.5: TableGen generates correct C++ classes

**Команда:**
```bash
# Проверяем, что заголовочные файлы сгенерировались
ls -lh build/include/circt/Dialect/Debug/DebugOps.h.inc
ls -lh build/include/circt/Dialect/Debug/DebugOps.cpp.inc

# Проверяем, что классы существуют
grep "class EnumDefOp" build/include/circt/Dialect/Debug/DebugOps.h.inc
grep "class ModuleInfoOp" build/include/circt/Dialect/Debug/DebugOps.h.inc
grep "class SubfieldOp" build/include/circt/Dialect/Debug/DebugOps.h.inc
```

**Ожидаемый результат:**
```
-rw-r--r-- 1 user user 45K Jan 18 05:40 DebugOps.h.inc
-rw-r--r-- 1 user user 38K Jan 18 05:40 DebugOps.cpp.inc
class EnumDefOp : public ::mlir::Op<...
class ModuleInfoOp : public ::mlir::Op<...
class SubfieldOp : public ::mlir::Op<...
```

---

#### AC-1.2.1: Invalid enum keys trigger verification error

**Тест:** `test/Dialect/Debug/ops-invalid.mlir`
```mlir
// RUN: circt-opt %s -split-input-file -verify-diagnostics

func.func @invalid_enum_key() {
  // expected-error @+1 {{enum value map keys must be numeric strings}}
  dbg.enumdef "BadEnum" {"abc" = "INVALID"}
  return
}
```

**Команда:**
```bash
./bin/llvm-lit -v ../test/Dialect/Debug/ops-invalid.mlir
```

**Ожидаемый результат:**
```
PASS: CIRCT :: Dialect/Debug/ops-invalid.mlir (1 of 1)
Testing Time: 0.12s
  Passed: 1
```

---

#### AC-1.3.1: Tests compile and run

**Команда:**
```bash
./bin/llvm-lit -v ../test/Dialect/Debug/ops.mlir
```

**Ожидаемый результат:**
```
PASS: CIRCT :: Dialect/Debug/ops.mlir (1 of 1)
```

**Если fail — смотрим детали:**
```bash
./bin/circt-opt ../test/Dialect/Debug/ops.mlir 2>&1 | head -n 30
```

---

### Phase 2: Verification Checklist

#### AC-2.1.1: UInt intrinsics lower correctly

**Тест:** `test/Dialect/FIRRTL/lower-intrinsics-debug.mlir` (Test 1)

**Команда:**
```bash
./bin/llvm-lit -v ../test/Dialect/FIRRTL/lower-intrinsics-debug.mlir
```

**Ожидаемый результат:**
```
// CHECK: dbg.variable "cpu.pc"
// CHECK-SAME: typeName = "UInt"
// CHECK-SAME: parameters = {width = 32}
// CHECK-NOT: @circt_debug_typeinfo
```

**Ручная проверка (если тест падает):**
```bash
./bin/circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' \
  ../test/Dialect/FIRRTL/lower-intrinsics-debug.mlir | \
  grep -A3 "dbg.variable"
```

---

#### AC-2.1.2: ChiselEnum intrinsics create dbg.enumdef

**Команда:**
```bash
./bin/circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' \
  ../test/Dialect/FIRRTL/lower-intrinsics-debug.mlir | \
  grep "dbg.enumdef"
```

**Ожидаемый результат:**
```mlir
%0 = dbg.enumdef "CpuState" {"0" = "IDLE", "1" = "FETCH", "2" = "DECODE"}
```

---

#### AC-2.1.6: Error handling for malformed intrinsics

**Тест:** `test/Dialect/FIRRTL/lower-intrinsics-debug-errors.mlir`
```mlir
// RUN: circt-opt %s -split-input-file -verify-diagnostics

firrtl.circuit "MissingTarget" {
  firrtl.intmodule @circt_debug_typeinfo_bad() attributes {
    intrinsic = "circt_debug_typeinfo",
    // Missing 'target' parameter!
    typeName = "UInt"
  }
  
  // expected-error @+1 {{circt_debug_typeinfo requires 'target' and 'typeName' parameters}}
  firrtl.module @MissingTarget() {}
}
```

**Команда:**
```bash
./bin/llvm-lit -v ../test/Dialect/FIRRTL/lower-intrinsics-debug-errors.mlir
```

---

### Phase 3: Verification Checklist

#### AC-3.1.2: JSON validates against schema

**Команда:**
```bash
# Генерируем JSON
./bin/firtool ../test/FIRRTL/debug-e2e.mlir --export-debug-info -o test.v

# Проверяем структуру
python3 << 'EOF'
import json
import sys

with open('hw-debug-info.json') as f:
    data = json.load(f)

# Validate schema
assert data['version'] == '1.0', "Wrong version"
assert 'modules' in data, "Missing modules"
assert len(data['modules']) > 0, "Empty modules"

for mod in data['modules']:
    assert 'name' in mod, "Module missing name"
    assert 'variables' in mod, "Module missing variables"
    for var in mod['variables']:
        assert 'name' in var, "Variable missing name"
        assert 'typeName' in var, "Variable missing typeName"

print("✓ JSON schema valid")
EOF
```

**Ожидаемый результат:**
```
✓ JSON schema valid
```

---

#### AC-3.2.5: All metadata preserved

**Команда:**
```bash
./bin/firtool ../test/FIRRTL/debug-e2e.mlir --export-debug-info -o test.v

# Проверяем, что всё на месте
jq '.modules[0].variables[] | select(.name == "cpu.pc")' hw-debug-info.json
```

**Ожидаемый результат:**
```json
{
  "name": "cpu.pc",
  "typeName": "UInt",
  "parameters": {"width": 32},
  "binding": "Reg",
  "rtlSignals": ["E2ETest_pc"]
}
```

---

## 🔄 Regression Prevention

### Перед каждым коммитом

```bash
# 1. Убедитесь, что изменения не сломали существующие тесты
cd build
ninja check-circt 2>&1 | tee test-results.log
grep "Passed:" test-results.log
# Ожидаемо: Passed: XXXX (число не должно уменьшаться!)

# 2. Проверьте, что новые тесты добавлены
find ../test -name "*.mlir" -newer ../THESIS_REQUIREMENTS.md

# 3. Убедитесь, что нет warning'ов в вашем коде
ninja 2>&1 | grep -i warning | grep -E "Debug|Intrinsic|Export"
# Ожидаемо: пустой вывод
```

---

## 📊 Progress Tracking

### Как отслеживать прогресс

**Метрики для диплома:**

```bash
# 1. Количество пройденных acceptance criteria
grep -c "\[x\]" THESIS_REQUIREMENTS.md
# Целевое значение: 25+ (из ~30)

# 2. Покрытие тестами
./bin/llvm-lit -v ../test/Dialect/Debug/ ../test/Dialect/FIRRTL/ | \
  grep "Passed:"
# Целевое значение: 15+ тестов

# 3. Размер кодовой базы
cloc lib/Dialect/Debug/ lib/Dialect/FIRRTL/Transforms/ lib/Conversion/ExportDebugInfo/
# Ожидаемо: ~1000-1500 LOC (без комментариев)

# 4. Документация
grep -c "///" include/circt/Dialect/Debug/DebugOps.td
# Целевое значение: 30+ строк Doxygen комментариев
```

---

## 🛠️ Build & Test Commands

### Initial Setup (Done Once)

```bash
# 1. Clone and initialize submodules
git clone https://github.com/Tako-San/circt.git
cd circt
git checkout feature/tywaves-intrinsics-lowering
git submodule update --init --recursive

# 2. Build LLVM/MLIR (takes ~30-60 min first time)
sudo apt install ccache lld g++
./utils/build-llvm.sh build install Release gcc g++ \
  -DLLVM_CCACHE_BUILD=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_PARALLEL_LINK_JOBS=2

# 3. Build CIRCT
mkdir -p build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm/install/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/install/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_ENABLE_LLD=ON
ninja
```

### Incremental Development Workflow

```bash
# After editing C++ files
cd build
ninja bin/circt-opt  # Rebuild only circt-opt

# Run specific test
./bin/llvm-lit -v ../test/Dialect/Debug/ops.mlir

# Run all Debug dialect tests
./bin/llvm-lit -v ../test/Dialect/Debug/

# Run all tests (slow, use sparingly)
ninja check-circt
```

### Debugging Failed Tests

```bash
# If test fails, run the command manually to see full output
./bin/circt-opt --pass-pipeline='builtin.module(...)' ../test/file.mlir

# Add --debug flag for verbose logging (if LLVM_DEBUG() calls exist)
./bin/circt-opt --debug --pass-pipeline='...' ../test/file.mlir
```

---

## 📋 Executive Summary

This document specifies the technical requirements for implementing debug metadata preservation in CIRCT, enabling source-level debugging for Chisel hardware designs. The implementation consists of three interconnected phases that must be completed in sequence.

**Core Objective:** Transform `circt_debug_typeinfo` FIRRTL intrinsics (generated by Chisel) into MLIR Debug Dialect operations, then export to `hw-debug-info.json` for consumption by waveform viewers (Tywaves, HGDB).

---

## 🎯 Goals and Non-Goals

### Goals

✅ **G1:** Extend CIRCT Debug Dialect with Chisel-specific operations  
✅ **G2:** Lower `circt_debug_typeinfo` intrinsics to Debug Dialect  
✅ **G3:** Export debug metadata to JSON format  
✅ **G4:** Preserve source location information (file + line)  
✅ **G5:** Support ChiselEnum value mappings  
✅ **G6:** Support Bundle/Vec hierarchy metadata  
✅ **G7:** Integrate with existing firtool pipeline  
✅ **G8:** Comprehensive test coverage (unit + integration)  

### Non-Goals

❌ **NG1:** VPI runtime implementation (Phase 3 of thesis)  
❌ **NG2:** Tywaves viewer modifications (separate project)  
❌ **NG3:** Chisel compiler changes (covered by Tako-San/chisel#1)  
❌ **NG4:** SystemVerilog bind statement generation  
❌ **NG5:** Performance optimization (acceptable overhead: <20%)  

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   CIRCT Debug Stack Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Chisel (Tako-San/chisel#1)                                     │
│    DebugInfo.annotate(signal, "name")                          │
│                ↓                                                │
│  FIRRTL IR                                                      │
│    intrinsic(circt_debug_typeinfo<...>, read(probe))           │
│                ↓                                                │
│  ╔═══════════════════════════════════════════════╗         │
│  ║ PHASE 1: Debug Dialect Extensions (THIS PR)      ║         │
│  ╠═══════════════════════════════════════════════╣         │
│  ║ DebugDialect.td                                   ║         │
│  ║   - dbg.enumdef  (ChiselEnum support)             ║         │
│  ║   - dbg.moduleinfo (module metadata)              ║         │
│  ║   - dbg.subfield (Bundle field tracking)          ║         │
│  ╚═══════════════════════════════════════════════╝         │
│                ↓                                                │
│  ╔═══════════════════════════════════════════════╗         │
│  ║ PHASE 2: Intrinsic Lowering (THIS PR)            ║         │
│  ╠═══════════════════════════════════════════════╣         │
│  ║ LowerIntrinsics.cpp                               ║         │
│  ║   - Parse circt_debug_typeinfo parameters         ║         │
│  ║   - Create dbg.variable + dbg.enumdef             ║         │
│  ║   - Extract signal from Probe                     ║         │
│  ║   - Preserve FileLineColLoc                       ║         │
│  ╚═══════════════════════════════════════════════╝         │
│                ↓                                                │
│  MLIR Debug Dialect                                             │
│    dbg.variable "cpu.pc" : UInt {width=32}                     │
│    dbg.enumdef "CpuState" {0="IDLE", 1="FETCH", ...}           │
│                ↓                                                │
│  ╔═══════════════════════════════════════════════╗         │
│  ║ PHASE 3: JSON Export (THIS PR)                    ║         │
│  ╠═══════════════════════════════════════════════╣         │
│  ║ ExportDebugInfo.cpp                               ║         │
│  ║   - Traverse dbg.* operations                     ║         │
│  ║   - Map MLIR values to RTL signal names           ║         │
│  ║   - Serialize to hw-debug-info.json               ║         │
│  ╚═══════════════════════════════════════════════╝         │
│                ↓                                                │
│  hw-debug-info.json                                             │
│    {"signals": [{"name":"cpu.pc", "type":"UInt", ...}]}        │
│                ↓                                                │
│  Waveform Viewers (Tywaves/HGDB)                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Phase 1: Debug Dialect Extensions

### REQ-1.1: Extend DebugDialect.td

**Priority:** 🔴 CRITICAL (must be done first)  
**Estimated Effort:** 3-4 hours  
**Source:** Based on rameloni's CIRCT fork  

#### Background Research

Rameloni's work demonstrates the need for additional Debug Dialect operations:
- **dbg.enumdef**: Store ChiselEnum value→name mappings
- **dbg.moduleinfo**: Preserve Chisel module hierarchy
- **dbg.subfield**: Track Bundle field access patterns

**Reference:** [rameloni/circt Wiki - Changes in CIRCT](https://github.com/rameloni/tywaves-chisel/wiki/Changes-in-CIRCT)

#### Implementation Tasks

**File:** `include/circt/Dialect/Debug/DebugOps.td`

```tablegen
//===----------------------------------------------------------------------===//
// EnumDefOp - Define enumeration types from hardware generators
//===----------------------------------------------------------------------===//

def EnumDefOp : DebugOp<"enumdef", [Pure]> {
  let summary = "Defines an enumeration type (ChiselEnum, SystemVerilog enum)";
  
  let description = [{
    Stores metadata about high-level enum types that lose their names during
    lowering. Maps enum integer values to source-level symbolic names.
    
    Example:
    ```mlir
    dbg.enumdef "CpuState" {
      "0" = "sIDLE",
      "1" = "sFETCH", 
      "2" = "sDECODE",
      "3" = "sEXECUTE"
    }
    ```
    
    Used by waveform viewers to display `state = sFETCH` instead of `state = 1`.
  }];
  
  let arguments = (ins
    StrAttr:$name,                          // Enum type name (e.g., "CpuState")
    DictionaryAttr:$value_map,              // Value→Name mapping
    OptionalAttr<StrAttr>:$source_file,     // Source file
    OptionalAttr<I32Attr>:$source_line      // Source line
  );
  
  let results = (outs);  // No runtime value, just metadata
  
  let assemblyFormat = [{
    $name $value_map (`loc` `(` $source_file^ `:` $source_line `)`)?
    attr-dict
  }];
  
  let builders = [
    OpBuilder<(ins "StringRef":$name, "DictionaryAttr":$valueMap)>
  ];
}

//===----------------------------------------------------------------------===//
// ModuleInfoOp - Preserve Chisel module hierarchy metadata
//===----------------------------------------------------------------------===//

def ModuleInfoOp : DebugOp<"moduleinfo", []> {
  let summary = "Stores Chisel module metadata (name, params, source location)";
  
  let description = [{
    Preserves information about Chisel modules that may be transformed or
    inlined during lowering. Enables debuggers to reconstruct the original
    module hierarchy.
    
    Example:
    ```mlir
    dbg.moduleinfo "Riscv5Stage" {
      params = {"xLen" = 32, "formalWidth" = 64},
      source = loc("Riscv.scala":42:7)
    }
    ```
  }];
  
  let arguments = (ins
    StrAttr:$module_name,
    OptionalAttr<DictionaryAttr>:$parameters,
    OptionalAttr<StrAttr>:$source_file,
    OptionalAttr<I32Attr>:$source_line
  );
  
  let assemblyFormat = [{
    $module_name (`params` `=` $parameters^)?
    (`loc` `(` $source_file^ `:` $source_line `)`)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SubfieldOp - Track Bundle/Struct field access
//===----------------------------------------------------------------------===//

def SubfieldOp : DebugOp<"subfield", [Pure]> {
  let summary = "Extract a field from a Bundle/Struct for debugging";
  
  let description = [{
    Represents field extraction from aggregate types (Chisel Bundle,
    SystemVerilog struct). Used to preserve type hierarchy in debug info.
    
    Example:
    ```mlir
    %bundle_dbg = dbg.variable "io" : !hw.struct<data: i32, valid: i1>
    %data_dbg = dbg.subfield %bundle_dbg["data"] : !hw.struct<data: i32, valid: i1> -> i32
    %valid_dbg = dbg.subfield %bundle_dbg["valid"] : !hw.struct<data: i32, valid: i1> -> i1
    ```
  }];
  
  let arguments = (ins
    AnyType:$input,
    StrAttr:$field_name
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $input `[` $field_name `]` attr-dict `:` type($input) `->` type($result)
  }];
}
```

#### Acceptance Criteria

- [ ] **AC-1.1.1:** `EnumDefOp` compiles without errors
- [ ] **AC-1.1.2:** `ModuleInfoOp` compiles without errors  
- [ ] **AC-1.1.3:** `SubfieldOp` compiles without errors
- [ ] **AC-1.1.4:** `ninja bin/circt-opt` succeeds
- [ ] **AC-1.1.5:** TableGen generates correct C++ classes

---

### REQ-1.2: Implement Operation Verifiers

**Priority:** 🟡 HIGH  
**Estimated Effort:** 1-2 hours  

**File:** `lib/Dialect/Debug/DebugOps.cpp`

```cpp
//===----------------------------------------------------------------------===//
// EnumDefOp verification
//===----------------------------------------------------------------------===//

LogicalResult EnumDefOp::verify() {
  auto valueMap = getValueMap();
  
  // Verify all keys are numeric strings
  for (auto [key, value] : valueMap) {
    auto keyStr = key.cast<StringAttr>().getValue();
    if (!std::all_of(keyStr.begin(), keyStr.end(), ::isdigit)) {
      return emitError("enum value map keys must be numeric strings, got: ")
             << keyStr;
    }
  }
  
  // Verify all values are strings
  for (auto [key, value] : valueMap) {
    if (!value.isa<StringAttr>()) {
      return emitError("enum value map values must be strings");
    }
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// SubfieldOp verification
//===----------------------------------------------------------------------===//

LogicalResult SubfieldOp::verify() {
  auto inputType = getInput().getType();
  
  // Verify input is a struct/bundle type
  if (!inputType.isa<hw::StructType>()) {
    return emitError("subfield input must be a struct type");
  }
  
  // Verify field exists
  auto structType = inputType.cast<hw::StructType>();
  if (!structType.hasField(getFieldName())) {
    return emitError("struct does not have field: ") << getFieldName();
  }
  
  return success();
}
```

#### Acceptance Criteria

- [ ] **AC-1.2.1:** Invalid enum keys trigger verification error
- [ ] **AC-1.2.2:** Invalid field names trigger verification error
- [ ] **AC-1.2.3:** All edge cases covered by tests

---

### REQ-1.3: Add Unit Tests

**Priority:** 🟡 HIGH  
**Estimated Effort:** 2 hours  

**File:** `test/Dialect/Debug/ops.mlir`

```mlir
// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK-LABEL: @enum_def_test
func.func @enum_def_test() {
  // CHECK: dbg.enumdef "CpuState" {"0" = "IDLE", "1" = "FETCH"}
  dbg.enumdef "CpuState" {
    "0" = "IDLE",
    "1" = "FETCH",
    "2" = "DECODE"
  }
  
  return
}

// CHECK-LABEL: @subfield_test
func.func @subfield_test(%arg0: !hw.struct<data: i32, valid: i1>) {
  %dbg = dbg.variable "bundle" : !hw.struct<data: i32, valid: i1>
  
  // CHECK: dbg.subfield %{{.*}}["data"] : !hw.struct<data: i32, valid: i1> -> i32
  %data = dbg.subfield %dbg["data"] : !hw.struct<data: i32, valid: i1> -> i32
  
  return
}
```

#### Acceptance Criteria

- [ ] **AC-1.3.1:** Tests compile and run
- [ ] **AC-1.3.2:** FileCheck patterns match output
- [ ] **AC-1.3.3:** Negative tests for invalid ops

---

[... REST OF DOCUMENT SAME AS BEFORE, TOO LONG TO INCLUDE ...]

---

## ✅ Checklist

### Before Starting

- [x] Review rameloni's CIRCT changes
- [x] Understand existing Debug Dialect
- [x] Set up CIRCT build environment
- [ ] Read MLIR Pass Infrastructure docs

### Phase 1 Complete

- [ ] EnumDefOp implemented
- [ ] ModuleInfoOp implemented
- [ ] SubfieldOp implemented
- [ ] Unit tests passing
- [ ] Code reviewed

### Phase 2 Complete

- [ ] Intrinsic lowering implemented
- [ ] Probe handling correct
- [ ] Enum parsing working
- [ ] Integration tests passing
- [ ] No regressions

### Phase 3 Complete

- [ ] JSON export implemented
- [ ] Schema validation passing
- [ ] E2E test with Chisel
- [ ] Documentation complete
- [ ] Ready for thesis demo

---

**Document Status:** 🟢 APPROVED FOR IMPLEMENTATION  
**Last Updated:** 2026-01-18  
**Next Review:** After Phase 1 completion