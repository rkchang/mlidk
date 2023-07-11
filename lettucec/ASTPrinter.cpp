#include "ASTPrinter.hpp"
#include "AST.hpp"
#include <iostream>

auto ASTPrinter::GetPrefix(std::any Context) -> std::string {
  int IndentAmount = std::any_cast<int>(Context) - 1;
  std::string s;
  if (IndentAmount > 0) {
    s = "" + std::string(IndentAmount, ' ');
  }
  s += "|";

  return s;
}

auto ASTPrinter::IncrContext(std::any Context) -> int {
  return std::any_cast<int>(Context) + 2;
}

auto ASTPrinter::visit(const LetExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "LetExpr: "
            << ".Name=" << Node.Name << "\n";
  auto NewContext = IncrContext(Context);
  Node.Value->accept(*this, NewContext);
  Node.Body->accept(*this, NewContext);
  return NULL;
}
auto ASTPrinter::visit(const BinaryExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "BinaryExpr: "
            << ".Operator="
            << " " << TokenOp::OpToStr(Node.Operator) << "\n";
  auto NewContext = IncrContext(Context);
  Node.Left->accept(*this, NewContext);
  Node.Right->accept(*this, NewContext);
  return NULL;
}
auto ASTPrinter::visit(const IntExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "IntExpr:"
            << " " << Node.Value << "\n";
  return NULL;
}
auto ASTPrinter::visit(const VarExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "VarExpr:"
            << " " << Node.Name << "\n";
  return NULL;
}