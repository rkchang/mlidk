#include "ASTPrinter.hpp"
#include "MLIR_gen.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "passes.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

std::optional<std::string> readFile(const std::string &Filename) {
  const std::ifstream IFS(Filename);
  if (!IFS) {
    std::cerr << "Error: could not open file '" << Filename << "'\n";
    return std::nullopt;
  }
  std::stringstream SrcBuffer;
  SrcBuffer << IFS.rdbuf();
  return SrcBuffer.str();
}

int main(int argc, char *argv[]) {
  if (argc <= 1 || argc > 2) {
    std::cerr << "Usage: ./mlidk <filename>"
              << "\n";
    return 1;
  }
  const std::string Filename(argv[1]);
  auto V = readFile(Filename);
  if (!V) {
    return 1;
  }
  auto Source = *V;

  // Print source
  std::cout << "Source:" << std::endl;
  std::cout << Source << std::endl;

  auto Lexr = Lexer(Source, Filename);
  auto Parsr = Parser(Lexr);
  auto AST = Parsr.parse();

  // Print out the AST
  std::cout << std::endl << "AST:" << std::endl;
  auto Printer = ASTPrinter();
  AST->accept(Printer, 0);

  // Generate MLIR
  mlir::MLIRContext Context;
  Context.getOrLoadDialect<mlir::arith::ArithDialect>();
  Context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto MLIRGenerator = MLIRGen(Context);
  AST->accept(MLIRGenerator, 0);
  mlir::OwningOpRef<mlir::ModuleOp> Module = MLIRGenerator.Module;

  // Print out the MLIR
  std::cout << std::endl << "MLIR:" << std::endl;
  Module->dump();

  if (failed(mlir::verify(Module->getOperation()))) {
    std::cout << "Failed to verify MLIR Module\n";
    return -1;
  }

  // Lower to LLVM
  auto PM = mlir::PassManager(&Context);
  applyPassManagerCLOptions(PM);
  PM.addPass(lettuce::createLowerToLLVMPass());

  if (mlir::failed(PM.run(*Module))) {
    return 4;
  }

  // Print out the LLVM dialect
  std::cout << std::endl << "LLVM Dialect:" << std::endl;
  Module->dump();

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(*Module->getContext());

  // Convert dialect to LLVM IR
  llvm::LLVMContext LlvmContext;
  auto LlvmModule =
      mlir::translateModuleToLLVMIR(Module->getOperation(), LlvmContext);
  if (!LlvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Print out the LLVM IR
  std::cout << std::endl << "LLVM IR:" << std::endl;
  LlvmModule->dump();
}