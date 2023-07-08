#include <iostream>
#include <llvm/IR/IRBuilder.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>

#include <mlidk/Dialect.h>

int main() {
  std::cout << "MLIRdk!";

  // Load dialect
  mlir::MLIRContext Context;
  Context.getOrLoadDialect<MlidkDialect>();

  // Create Builder and Module
  mlir::OpBuilder Builder(&Context);
  mlir::ModuleOp Module = mlir::ModuleOp::create(Builder.getUnknownLoc());

  // Attempt to create an Operation
  mlir::Type Type = mlir::IntegerType::get(&Context, 32);
  mlir::IntegerAttr Attr = mlir::IntegerAttr();
  Builder.create<ConstantOp>(Builder.getUnknownLoc(), Type, Attr);

  Module.dump();
}