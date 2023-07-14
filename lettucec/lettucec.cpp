#include "ASTPrinter.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

#include "Romaine/RomaineOps.h"

std::optional<std::string> readFile(const std::string &Filename) {
  std::ifstream IFS(Filename);
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
  std::string Filename(argv[1]);
  auto V = readFile(Filename);
  if (!V) {
    return 1;
  }
  auto Source = *V;

  auto Lexr = Lexer(Source, Filename);
  auto Parsr = Parser(Lexr);
  auto AST = Parsr.parse();
  auto Printer = ASTPrinter();
  AST->accept(Printer, 0);

  auto fooOp = mlir::romaine::FooOp();
}