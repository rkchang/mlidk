#include "ASTPrinter.hpp"
#include "AST.fwd.hpp"
#include "AST.hpp"
#include <iostream>
#include <string>

auto getType(const Expr &Exp) -> std::string {
  auto Ty = Exp.Ty;
  if (Ty) {
    return Ty->toString();
  }
  return "Unknown";
}

auto ASTPrinter::GetPrefix(std::any Context) -> std::string {
  const int IndentAmount = std::any_cast<int>(Context) - 1;
  std::string S;
  if (IndentAmount > 0) {
    S = "" + std::string(IndentAmount, ' ');
  }
  S += "|";

  return S;
}

auto ASTPrinter::IncrContext(std::any Context) -> int {
  return std::any_cast<int>(Context) + 2;
}

auto ASTPrinter::visit(const RootNode &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "RootNode:"
            << "\n";
  auto NewContext = IncrContext(Context);
  Node.Exp->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const LetExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "LetExpr: "
            << ".Name=" << Node.Name << " .Type=" << getType(Node) << "\n";
  auto NewContext = IncrContext(Context);
  Node.Value->accept(*this, NewContext);
  Node.Body->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const IfExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "IfExpr:"
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = IncrContext(Context);
  Node.Condition->accept(*this, NewContext);
  Node.TrueBranch->accept(*this, NewContext);
  Node.FalseBranch->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const BinaryExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "BinaryExpr: "
            << ".Operator="
            << " " << TokenOp::OpToStr(Node.Operator)
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = IncrContext(Context);
  Node.Left->accept(*this, NewContext);
  Node.Right->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const UnaryExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "UnaryExpr: "
            << ".Operator="
            << " " << TokenOp::OpToStr(Node.Operator)
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = IncrContext(Context);
  Node.Right->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const IntExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "IntExpr:"
            << " " << Node.Value << " .Type=" << getType(Node) << "\n";
  return NULL;
}

auto ASTPrinter::visit(const BoolExpr &Node, std::any Context) -> std::any {
  const auto *Value = Node.Value ? "true" : "false";
  std::cout << GetPrefix(Context) << "BoolExpr:"
            << " " << Value << " .Type=" << getType(Node) << "\n";
  return NULL;
}

auto ASTPrinter::visit(const VarExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "VarExpr:"
            << " " << Node.Name << " .Type=" << getType(Node) << "\n";
  return NULL;
}

auto ASTPrinter::visit(const CallExpr &Node, std::any Context) -> std::any {
  std::cout << GetPrefix(Context) << "CallExpr:"
            << " " << Node.FuncName << "\n";
  auto NewContext = IncrContext(Context);
  for (const auto &Arg : Node.Args) {
    Arg->accept(*this, NewContext);
  }
  return NULL;
}
