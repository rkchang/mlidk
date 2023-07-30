#pragma once

#include "AST.fwd.hpp"
#include "ASTVisitor.hpp"
#include <string>
class ASTPrinter : public ASTVisitor {
public:
  auto visit(const RootNode &Node, std::any Context) -> std::any override;

  auto visit(const DefExpr &Node, std::any Context) -> std::any override;
  auto visit(const LetExpr &Node, std::any Context) -> std::any override;
  auto visit(const IfExpr &Node, std::any Context) -> std::any override;
  auto visit(const BinaryExpr &Node, std::any Context) -> std::any override;
  auto visit(const UnaryExpr &Node, std::any Context) -> std::any override;
  auto visit(const IntExpr &Node, std::any Context) -> std::any override;
  auto visit(const BoolExpr &Node, std::any Context) -> std::any override;
  auto visit(const VarExpr &Node, std::any Context) -> std::any override;
  auto visit(const CallExpr &Node, std::any Context) -> std::any override;
  auto visit(const FuncExpr &Node, std::any Context) -> std::any override;
};
