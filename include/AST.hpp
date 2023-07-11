#pragma once

#include "ASTVisitor.hpp"
#include "lexer.hpp"
#include <any>
#include <memory>
#include <string>

class ASTNode {
public:
  virtual ~ASTNode() = default;
  virtual std::any accept(ASTVisitor &Visitor, std::any Context) const = 0;
};

class Expr : public ASTNode {};

class LetExpr : public Expr {
public:
  std::string Name;
  std::unique_ptr<Expr> Value;
  std::unique_ptr<Expr> Body;

  LetExpr(std::string Name, std::unique_ptr<Expr> Value,
          std::unique_ptr<Expr> Body);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class BinaryExpr : public Expr {
public:
  std::unique_ptr<Expr> Left;
  TokenOp::OpType Operator;
  std::unique_ptr<Expr> Right;

  BinaryExpr(std::unique_ptr<Expr> Left, TokenOp::OpType Operator,
             std::unique_ptr<Expr> Right);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class IntExpr : public Expr {
public:
  int Value;

  IntExpr(int Value);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class VarExpr : public Expr {
public:
  std::string Name;

  VarExpr(std::string Name);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};
