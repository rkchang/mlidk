#ifndef MLIDK_DIALECT
#define MLIDK_DIALECT

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

/**
 * A Dialect for Mlidk
 */
class MlidkDialect : public mlir::Dialect {
public:
  explicit MlidkDialect(mlir::MLIRContext *Ctx);

  static llvm::StringRef getDialectNamespace() { return "mlidk"; }

  void initialize();
};

/**
 * An operation representing a constant value
 */
class ConstantOp
    : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroOperands,
                      mlir::OpTrait::OneResult,
                      mlir::OpTrait::OneTypedResult<mlir::IntegerType>::Impl> {

public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "mlidk.constant"; }

  mlir::IntegerAttr getValue();

  mlir::LogicalResult verifyInvariants();

  static void build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                    mlir::Type Result, mlir::IntegerAttr Value);
};

#endif