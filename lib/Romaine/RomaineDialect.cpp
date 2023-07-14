//===- RomaineDialect.cpp - Romaine dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Romaine/RomaineDialect.h"
#include "Romaine/RomaineOps.h"

using namespace mlir;
using namespace mlir::romaine;

#include "Romaine/RomaineOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Romaine dialect.
//===----------------------------------------------------------------------===//

void RomaineDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Romaine/RomaineOps.cpp.inc"
      >();
}
