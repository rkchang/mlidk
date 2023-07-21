#include "AST.hpp"
#include "AST.fwd.hpp"
#include <cassert>

LetExpr::LetExpr(Location Loc, std::string Name, std::unique_ptr<Expr> Value,
                 std::unique_ptr<Expr> Body)
    : Expr(Loc), Name(Name), Value(std::move(Value)), Body(std::move(Body)) {
  assert(this->Value != nullptr);
  assert(this->Body != nullptr);
}

auto LetExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

IfExpr::IfExpr(Location Loc, std::unique_ptr<Expr> Condition,
               std::unique_ptr<Expr> TrueBranch,
               std::unique_ptr<Expr> FalseBranch)
    : Expr(Loc), Condition(std::move(Condition)),
      TrueBranch(std::move(TrueBranch)), FalseBranch(std::move(FalseBranch)) {
  assert(this->Condition != nullptr);
  assert(this->TrueBranch != nullptr);
  assert(this->FalseBranch != nullptr);
}

auto IfExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

BinaryExpr::BinaryExpr(Location Loc, std::unique_ptr<Expr> Left,
                       TokenOp::OpType Operator, std::unique_ptr<Expr> Right)
    : Expr(Loc), Left(std::move(Left)), Operator(Operator),
      Right(std::move(Right)) {
  assert(this->Left != nullptr);
  assert(this->Right != nullptr);
}

auto BinaryExpr::accept(ASTVisitor &Visitor, std::any Context) const
    -> std::any {
  return Visitor.visit(*this, Context);
}

UnaryExpr::UnaryExpr(Location Loc, TokenOp::OpType Operator,
                     std::unique_ptr<Expr> Right)
    : Expr(Loc), Operator(Operator), Right(std::move(Right)) {
  assert(this->Right != nullptr);
}

auto UnaryExpr::accept(ASTVisitor &Visitor, std::any Context) const
    -> std::any {
  return Visitor.visit(*this, Context);
}

IntExpr::IntExpr(Location Loc, int Value) : Expr(Loc), Value(Value) {}

auto IntExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

BoolExpr::BoolExpr(Location Loc, bool Value) : Expr(Loc), Value(Value) {}

auto BoolExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

VarExpr::VarExpr(Location Loc, std::string Name) : Expr(Loc), Name(Name) {}

auto VarExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}
