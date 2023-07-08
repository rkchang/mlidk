#include "AST.hpp"

LetExpr::LetExpr(std::string Name, std::unique_ptr<Expr> Value,
                 std::unique_ptr<Expr> Body)
    : Name(Name), Value(std::move(Value)), Body(std::move(Body)) {
  assert(this->Value != nullptr);
  assert(this->Body != nullptr);
}

auto LetExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

BinaryExpr::BinaryExpr(std::unique_ptr<Expr> Left, TokenOp::OpType Operator,
                       std::unique_ptr<Expr> Right)
    : Left(std::move(Left)), Operator(Operator), Right(std::move(Right)) {
  assert(this->Left != nullptr);
  assert(this->Right != nullptr);
}

auto BinaryExpr::accept(ASTVisitor &Visitor, std::any Context) const
    -> std::any {
  return Visitor.visit(*this, Context);
}

IntExpr::IntExpr(int Value) : Value(Value) {}

auto IntExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

VarExpr::VarExpr(std::string Name) : Name(Name) {}

auto VarExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}
