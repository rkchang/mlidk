#pragma once

#include "ASTVisitor.hpp"
#include "lexer.hpp"
#include "types.hpp"

#include <any>
#include <memory>
#include <optional>
#include <string>

struct Location {
  std::string Filename; // TODO: make shared_ptr
  int Line;
  int Column;
};

class ASTNode {
public:
  Location Loc;

  ASTNode(Location Loc) : Loc(Loc) {}
  virtual ~ASTNode() = default;
  virtual std::any accept(ASTVisitor &Visitor, std::any Context) const = 0;
};

enum class ExprKind { LET, IF, BIN_OP, UN_OP, INT, BOOL, VAR };

class Expr : public ASTNode {
public:
  const ExprKind Kind;
  std::optional<Type> Ty;
  Expr(Location Loc, ExprKind Kind) : ASTNode(Loc), Kind(Kind) {}
};

class RootNode : public ASTNode {
public:
  std::unique_ptr<Expr> Exp;

  RootNode(Location Loc, std::unique_ptr<Expr> Exp);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class LetExpr : public Expr {
public:
  std::string Name;
  std::unique_ptr<Expr> Value;
  std::unique_ptr<Expr> Body;

  LetExpr(Location Loc, std::string Name, std::unique_ptr<Expr> Value,
          std::unique_ptr<Expr> Body);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class IfExpr : public Expr {
public:
  std::unique_ptr<Expr> Condition;
  std::unique_ptr<Expr> TrueBranch;
  std::unique_ptr<Expr> FalseBranch;

  IfExpr(Location Loc, std::unique_ptr<Expr> Condition,
         std::unique_ptr<Expr> TrueBranch, std::unique_ptr<Expr> FalseBranch);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class BinaryExpr : public Expr {
public:
  std::unique_ptr<Expr> Left;
  TokenOp::OpType Operator;
  std::unique_ptr<Expr> Right;

  BinaryExpr(Location Loc, std::unique_ptr<Expr> Left, TokenOp::OpType Operator,
             std::unique_ptr<Expr> Right);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class UnaryExpr : public Expr {
public:
  TokenOp::OpType Operator;
  std::unique_ptr<Expr> Right;

  UnaryExpr(Location Loc, TokenOp::OpType Operator,
            std::unique_ptr<Expr> Right);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class IntExpr : public Expr {
public:
  int Value;

  IntExpr(Location Loc, int Value);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class BoolExpr : public Expr {
public:
  bool Value;

  BoolExpr(Location Loc, bool Value);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class VarExpr : public Expr {
public:
  std::string Name;

  VarExpr(Location Loc, std::string Name);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};

class CallExpr : public Expr {
public:
  std::string FuncName;
  std::vector<std::unique_ptr<Expr>> Args;

  CallExpr(Location Loc, std::string FuncName,
           std::vector<std::unique_ptr<Expr>> Args);
  auto accept(ASTVisitor &Visitor, std::any Context) const -> std::any override;
};
