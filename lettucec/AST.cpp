#include "AST.hpp"
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

IntExpr::IntExpr(Location Loc, int Value) : Expr(Loc), Value(Value) {}

auto IntExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

VarExpr::VarExpr(Location Loc, std::string Name) : Expr(Loc), Name(Name) {}

auto VarExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}
