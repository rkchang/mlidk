#pragma once

#include "AST.fwd.hpp"
#include <any>

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;

  // Root
  virtual std::any visit(RootNode &Node, std::any Context) = 0;

  // Expressions
  virtual std::any visit(DefExpr &Node, std::any Context) = 0;
  virtual std::any visit(LetExpr &Node, std::any Context) = 0;
  virtual std::any visit(IfExpr &Node, std::any Context) = 0;
  virtual std::any visit(BinaryExpr &Node, std::any Context) = 0;
  virtual std::any visit(UnaryExpr &Node, std::any Context) = 0;
  virtual std::any visit(IntExpr &Node, std::any Context) = 0;
  virtual std::any visit(BoolExpr &Node, std::any Context) = 0;
  virtual std::any visit(VarExpr &Node, std::any Context) = 0;
  virtual std::any visit(CallExpr &Node, std::any Context) = 0;
  virtual std::any visit(FuncExpr &Node, std::any Context) = 0;
};

// TODO: Add default visiting
