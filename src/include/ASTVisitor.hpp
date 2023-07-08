#pragma once

#include "AST.fwd.hpp"
#include <any>

class ASTVisitor {
public:
  virtual std::any visit(const LetExpr &Node, std::any Context) = 0;
  virtual std::any visit(const BinaryExpr &Node, std::any Context) = 0;
  virtual std::any visit(const IntExpr &Node, std::any Context) = 0;
  virtual std::any visit(const VarExpr &Node, std::any Context) = 0;
};

// TODO: Add default visiting