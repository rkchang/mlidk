#include "ASTPrinter.hpp"
#include "MLIR_gen.hpp"
#include "lexer.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "parser.hpp"

#include <fstream>
#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <sstream>

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

  std::cout << Source << std::endl;

  auto Lexr = Lexer(Source, Filename);
  auto Parsr = Parser(Lexr);
  auto AST = Parsr.parse();

  // Print out the AST
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
  Module->dump();
}