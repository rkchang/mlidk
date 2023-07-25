#pragma once

#include "AST.fwd.hpp"
#include <any>

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;

  // Root
  virtual std::any visit(const RootNode &Node, std::any Context) = 0;

  // Expressions
  virtual std::any visit(const LetExpr &Node, std::any Context) = 0;
  virtual std::any visit(const IfExpr &Node, std::any Context) = 0;
  virtual std::any visit(const BinaryExpr &Node, std::any Context) = 0;
  virtual std::any visit(const UnaryExpr &Node, std::any Context) = 0;
  virtual std::any visit(const IntExpr &Node, std::any Context) = 0;
  virtual std::any visit(const BoolExpr &Node, std::any Context) = 0;
  virtual std::any visit(const VarExpr &Node, std::any Context) = 0;
  virtual std::any visit(const CallExpr &Node, std::any Context) = 0;
  virtual std::any visit(const FuncExpr &Node, std::any Context) = 0;
};

// TODO: Add default visiting