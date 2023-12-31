#include "AST.hpp"

#include <cassert>

RootNode::RootNode(Location Loc, std::unique_ptr<Expr> Exp)
    : ASTNode(Loc), Exp(std::move(Exp)) {
  assert(this->Exp != nullptr);
}

auto RootNode::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

DefBinder::DefBinder(Location Loc, std::string Name,
                     std::vector<std::pair<std::string, Type>> Params,
                     Type ReturnType, std::unique_ptr<Expr> Body)
    : Loc(Loc), Name(Name), Params(Params), ReturnType(ReturnType),
      Body(std::move(Body)), Ty(nullptr) {}

DefExpr::DefExpr(Location Loc, std::vector<DefBinder> Definitions,
                 std::unique_ptr<Expr> Body)
    : Expr(Loc, ExprKind::DEF), Definitions(std::move(Definitions)),
      Body(std::move(Body)) {
  assert(this->Definitions.size() >= 1 &&
         "Must define at least one definition");
  assert(this->Body != nullptr);
}

auto DefExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

LetExpr::LetExpr(Location Loc, std::string Name, std::unique_ptr<Expr> Value,
                 std::unique_ptr<Expr> Body)
    : Expr(Loc, ExprKind::LET), Name(Name), Value(std::move(Value)),
      Body(std::move(Body)) {
  assert(this->Value != nullptr);
  assert(this->Body != nullptr);
}

auto LetExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

IfExpr::IfExpr(Location Loc, std::unique_ptr<Expr> Condition,
               std::unique_ptr<Expr> TrueBranch,
               std::unique_ptr<Expr> FalseBranch)
    : Expr(Loc, ExprKind::IF), Condition(std::move(Condition)),
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
    : Expr(Loc, ExprKind::BIN_OP), Left(std::move(Left)), Operator(Operator),
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
    : Expr(Loc, ExprKind::UN_OP), Operator(Operator), Right(std::move(Right)) {
  assert(this->Right != nullptr);
}

auto UnaryExpr::accept(ASTVisitor &Visitor, std::any Context) const
    -> std::any {
  return Visitor.visit(*this, Context);
}

IntExpr::IntExpr(Location Loc, int Value)
    : Expr(Loc, ExprKind::INT), Value(Value) {}

auto IntExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

BoolExpr::BoolExpr(Location Loc, bool Value)
    : Expr(Loc, ExprKind::BOOL), Value(Value) {}

auto BoolExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

VarExpr::VarExpr(Location Loc, std::string Name)
    : Expr(Loc, ExprKind::VAR), Name(Name) {}

auto VarExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

CallExpr::CallExpr(Location Loc, std::unique_ptr<Expr> Func,
                   std::vector<std::unique_ptr<Expr>> Args)
    : Expr(Loc, ExprKind::CALL), Func(std::move(Func)), Args(std::move(Args)) {}

auto CallExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}

FuncExpr::FuncExpr(Location Loc,
                   std::vector<std::pair<std::string, Type>> Params,
                   std::unique_ptr<Expr> Body)
    : Expr(Loc, ExprKind::FUNC), Params(Params), Body(std::move(Body)) {}

auto FuncExpr::accept(ASTVisitor &Visitor, std::any Context) const -> std::any {
  return Visitor.visit(*this, Context);
}