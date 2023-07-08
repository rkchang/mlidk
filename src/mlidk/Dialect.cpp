#include <iostream>

#include <mlir/IR/Builders.h>

#include <mlidk/Dialect.h>

auto MlidkDialect::initialize() -> void {
  // Add operations to dialect
  addOperations<ConstantOp>();
}

auto ConstantOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                       mlir::Type Result, mlir::IntegerAttr Value) -> void {
  std::cout << "building ConstantOp..." << std::endl;
}
