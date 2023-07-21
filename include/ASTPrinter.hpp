#pragma once

#include "ASTVisitor.hpp"
#include <string>
class ASTPrinter : public ASTVisitor {
public:
  auto visit(const LetExpr &Node, std::any Context) -> std::any override;
  auto visit(const IfExpr &Node, std::any Context) -> std::any override;
  auto visit(const BinaryExpr &Node, std::any Context) -> std::any override;
  auto visit(const UnaryExpr &Node, std::any Context) -> std::any override;
  auto visit(const IntExpr &Node, std::any Context) -> std::any override;
  auto visit(const BoolExpr &Node, std::any Context) -> std::any override;
  auto visit(const VarExpr &Node, std::any Context) -> std::any override;

private:
  /*
   * Get the indent string for the current context.
   */
  auto GetPrefix(std::any Context) -> std::string;
  /*
   * Increment the context by one.
   */
  auto IncrContext(std::any Context) -> int;
};
