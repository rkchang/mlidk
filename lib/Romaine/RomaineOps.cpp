//===- RomaineOps.cpp - Romaine dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Romaine/RomaineOps.h"
#include "Romaine/RomaineDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Romaine/RomaineOps.cpp.inc"

//void mlir::romaine::ConstantOp::build(mlir::OpBuilder &builder,
//                                      mlir::OperationState &state, int value) {
//  mlir::MLIRContext Context; // TODO: What???
//  auto dataType = IntegerType::get(&Context, 32, IntegerType::Signed);
//  auto dataAttribute = IntegerAttr::get(dataType, value);
//  ConstantOp::build(builder, state, dataType, dataAttribute);
//}

// void mlir::romaine::BinOp::build(mlir::OpBuilder &builder,
//                                  mlir::OperationState &state, mlir::Value
//                                  lhs, mlir::Value rhs) {
//   state.addTypes(builder.getI32Type());
//   state.addOperands({lhs, rhs});
// }