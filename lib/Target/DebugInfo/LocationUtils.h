//===- LocationUtils.h - Shared loc helpers for DI emitters ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers shared by EmitHGLDD and EmitUHDI for picking source/HDL locations
// out of a (potentially nested) Location tree.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_LIB_TARGET_DEBUGINFO_LOCATIONUTILS_H
#define CIRCT_LIB_TARGET_DEBUGINFO_LOCATIONUTILS_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace circt {
namespace debuginfo {

/// Collect FileLineColLoc values reachable past exactly `level`
/// NameLoc("emitted") / FusedLoc{metadata="verilogLocations"} wrappers.
/// Each such wrapper consumes one level; once at level 0 we stop on the
/// next wrapper. Setting level=0 collects source ("HGL") locations;
/// level=1 collects emitted ("HDL") locations.
void findLocations(mlir::Location loc, unsigned level,
                   llvm::SmallVectorImpl<mlir::FileLineColLoc> &locs);

/// `fs::exists` with a per-emitter cache to avoid re-statting the same
/// files thousands of times during a per-op location walk.
bool cachedExists(llvm::StringRef path, llvm::StringMap<bool> &cache);

/// Best source / HDL location (prefers non-`.fir` over `.fir`). Returns
/// null if no location matches. If `onlyExisting` is true, drops paths
/// that don't exist; pass a non-null `existsCache` to memoise the
/// existence checks across calls.
mlir::FileLineColLoc bestLocation(mlir::Location loc, bool emitted,
                                  bool onlyExisting,
                                  llvm::StringMap<bool> *existsCache = nullptr);

} // namespace debuginfo
} // namespace circt

#endif // CIRCT_LIB_TARGET_DEBUGINFO_LOCATIONUTILS_H
