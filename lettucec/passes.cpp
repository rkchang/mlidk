#include "passes.hpp"

#include "mlir/Pass/Pass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include <mlir/Dialect/SCF/IR/SCF.h>

struct LettuceToLLVMLoweringPass
    : public mlir::PassWrapper<LettuceToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LettuceToLLVMLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &Registry) const override {
    Registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;
};

auto LettuceToLLVMLoweringPass::runOnOperation() -> void {
  auto Target = mlir::LLVMConversionTarget(getContext());
  Target.addLegalOp<mlir::ModuleOp>();

  auto TypeConverter = mlir::LLVMTypeConverter(&getContext());

  auto Patterns = mlir::RewritePatternSet(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(TypeConverter, Patterns);
  mlir::populateSCFToControlFlowConversionPatterns(Patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(TypeConverter,
                                                        Patterns);
  mlir::populateFuncToLLVMConversionPatterns(TypeConverter, Patterns);

  auto Module = getOperation();
  if (failed(applyFullConversion(Module, Target, std::move(Patterns)))) {
    signalPassFailure();
  }
}

namespace lettuce {
auto createLowerToLLVMPass() -> std::unique_ptr<mlir::Pass> {
  return std::make_unique<LettuceToLLVMLoweringPass>();
}
} // namespace lettuce