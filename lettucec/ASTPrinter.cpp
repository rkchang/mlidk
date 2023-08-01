#include "ASTPrinter.hpp"
#include "AST.fwd.hpp"
#include "AST.hpp"
#include "ASTVisitor.hpp"
#include "types.hpp"

#include <iostream>
#include <string>

auto getType(const Expr &Exp) -> std::string {
  auto Ty = Exp.Ty;
  if (Ty) {
    return Ty->toString();
  }
  return "Unknown";
}

auto getPrefix(std::any Context) -> std::string {
  const int IndentAmount = std::any_cast<int>(Context) - 1;
  std::string S;
  if (IndentAmount > 0) {
    S = "" + std::string(IndentAmount, ' ');
  }
  S += "|";

  return S;
}

auto incrContext(std::any Context) -> int {
  return std::any_cast<int>(Context) + 2;
}

auto printDefBinder(const DefBinder &Binder, int Context) -> void {
  std::cout << getPrefix(Context) << "Binder: "
            << ".Name= " << Binder.Name << "\n";
  // TODO: Print return type
  auto NewContext = incrContext(Context);
  for (const auto &Param : Binder.Params) {
    auto Ty = Param.second;
    std::cout << getPrefix(NewContext) << ".Param= " << Param.first << " : "
              << Ty.toString() << "\n";
  }
  auto Printer = ASTPrinter();
  std::cout << getPrefix(Context) << ".Body=";
  Binder.Body->accept(Printer, Context);
}

//

auto ASTPrinter::visit(const RootNode &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "RootNode:"
            << "\n";
  auto NewContext = incrContext(Context);
  Node.Exp->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const DefExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "DefExpr:\n";
  auto NewContext = incrContext(Context);
  for (auto &Binder : Node.Definitions) {
    printDefBinder(Binder, NewContext);
  }
  Node.Body->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const LetExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "LetExpr: "
            << ".Name=" << Node.Name << " .Type=" << getType(Node) << "\n";
  auto NewContext = incrContext(Context);
  Node.Value->accept(*this, NewContext);
  Node.Body->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const IfExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "IfExpr:"
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = incrContext(Context);
  Node.Condition->accept(*this, NewContext);
  Node.TrueBranch->accept(*this, NewContext);
  Node.FalseBranch->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const BinaryExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "BinaryExpr: "
            << ".Operator="
            << " " << TokenOp::OpToStr(Node.Operator)
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = incrContext(Context);
  Node.Left->accept(*this, NewContext);
  Node.Right->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const UnaryExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "UnaryExpr: "
            << ".Operator="
            << " " << TokenOp::OpToStr(Node.Operator)
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = incrContext(Context);
  Node.Right->accept(*this, NewContext);
  return NULL;
}

auto ASTPrinter::visit(const IntExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "IntExpr:"
            << " " << Node.Value << " .Type=" << getType(Node) << "\n";
  return NULL;
}

auto ASTPrinter::visit(const BoolExpr &Node, std::any Context) -> std::any {
  const auto *Value = Node.Value ? "true" : "false";
  std::cout << getPrefix(Context) << "BoolExpr:"
            << " " << Value << " .Type=" << getType(Node) << "\n";
  return NULL;
}

auto ASTPrinter::visit(const VarExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "VarExpr:"
            << " " << Node.Name << " .Type=" << getType(Node) << "\n";
  return NULL;
}

auto ASTPrinter::visit(const CallExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "CallExpr:"
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = incrContext(Context);
  Node.Func->accept(*this, NewContext);
  for (const auto &Arg : Node.Args) {
    Arg->accept(*this, NewContext);
  }
  return NULL;
}

auto ASTPrinter::visit(const FuncExpr &Node, std::any Context) -> std::any {
  std::cout << getPrefix(Context) << "FuncExpr:"
            << " .Type=" << getType(Node) << "\n";
  auto NewContext = incrContext(Context);
  for (const auto &Param : Node.Params) {
    auto Ty = Param.second;
    std::cout << getPrefix(NewContext) << ".Param= " << Param.first << " : "
              << Ty.toString() << "\n";
  }
  std::cout << getPrefix(NewContext) << ".Body= "
            << "\n";
  Node.Body->accept(*this, NewContext);
  return NULL;
}
