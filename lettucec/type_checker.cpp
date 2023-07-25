#include "type_checker.hpp"
#include "AST.fwd.hpp"
#include "types.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

auto typeBinaryOperator(TypeCtx Ctx, TokenOp::OpType Operator, Expr &Lhs,
                        Expr &Rhs) -> std::shared_ptr<Type> {
  switch (Operator) {
    // Arithmetic
  case TokenOp::OpType::ADD:
  case TokenOp::OpType::MINUS:
  case TokenOp::OpType::MUL:
  case TokenOp::OpType::DIV:
    typeCheck(Ctx, Lhs, Int32T);
    typeCheck(Ctx, Rhs, Int32T);
    return Int32T;

  // Comparisson
  case TokenOp::OpType::LT:
  case TokenOp::OpType::LE:
  case TokenOp::OpType::GT:
  case TokenOp::OpType::GE:
    typeCheck(Ctx, Lhs, Int32T);
    typeCheck(Ctx, Rhs, Int32T);
    return BoolT;

  // Boolean
  case TokenOp::OpType::AND:
  case TokenOp::OpType::OR:
    typeCheck(Ctx, Lhs, BoolT);
    typeCheck(Ctx, Rhs, BoolT);
    return BoolT;

  // Equality
  case TokenOp::OpType::EQ:
  case TokenOp::OpType::NE: {
    auto LhsTy = typeInfer(Ctx, Lhs);
    typeCheck(Ctx, Rhs, LhsTy);
    return BoolT;
  }

  // Unary
  case TokenOp::OpType::NOT:
    throw TypeError(Lhs.Loc, "Unsupported operation");
  }
}

auto typeUnaryOperator(TypeCtx Ctx, TokenOp::OpType Operator, Expr &Rhs)
    -> std::shared_ptr<Type> {
  switch (Operator) {
  case TokenOp::OpType::NOT:
    typeCheck(Ctx, Rhs, BoolT);
    return BoolT;

  case TokenOp::OpType::ADD:
  case TokenOp::OpType::MINUS:
  case TokenOp::OpType::MUL:
  case TokenOp::OpType::DIV:
  case TokenOp::OpType::EQ:
  case TokenOp::OpType::NE:
  case TokenOp::OpType::LT:
  case TokenOp::OpType::LE:
  case TokenOp::OpType::GT:
  case TokenOp::OpType::GE:
  case TokenOp::OpType::AND:
  case TokenOp::OpType::OR:
    throw TypeError(Rhs.Loc, "Unsupported operation");
  }
}

auto typeCheck(TypeCtx Ctx, Expr &Exp, std::shared_ptr<Type> Expected) -> void {
  auto Actual = typeInfer(Ctx, Exp);
  // Compare underlying Type value!
  if (*Actual != *Expected) {
    throw TypeError(Exp.Loc, "Expected " + Expected->toString() + ", but got " +
                                 Actual->toString());
  }
  Exp.Ty = Actual;
}

auto typeInfer(TypeCtx Ctx, Expr &Exp) -> std::shared_ptr<Type> {
  switch (Exp.Kind) {
  case ExprKind::INT: {
    Exp.Ty = Int32T;
    return Int32T;
  }
  case ExprKind::BOOL: {
    Exp.Ty = BoolT;
    return BoolT;
  }
  case ExprKind::VAR: {
    auto *E = static_cast<VarExpr *>(&Exp);
    if (Ctx.contains(E->Name)) {

      auto Ty = Ctx[E->Name];
      Exp.Ty = Ty;
      return Ty;
    }
    throw TypeError(Exp.Loc, "Undefined variable '" + E->Name + "'");
  }
  case ExprKind::LET: {
    auto *E = static_cast<LetExpr *>(&Exp);
    auto ValueTy = typeInfer(Ctx, *(E->Value));
    Ctx[E->Name] = ValueTy;
    auto BodyTy = typeInfer(Ctx, *(E->Body));
    Ctx.erase(E->Name);
    Exp.Ty = BodyTy;
    return BodyTy;
  }
  case ExprKind::IF: {
    auto *E = static_cast<IfExpr *>(&Exp);
    typeCheck(Ctx, *(E->Condition), BoolT);
    auto TrueBranchTy = typeInfer(Ctx, *(E->TrueBranch));
    typeCheck(Ctx, *(E->FalseBranch), TrueBranchTy);
    Exp.Ty = TrueBranchTy;
    return TrueBranchTy;
  }
  case ExprKind::BIN_OP: {
    auto *E = static_cast<BinaryExpr *>(&Exp);
    auto Ty = typeBinaryOperator(Ctx, E->Operator, *(E->Left), *(E->Right));
    Exp.Ty = Ty;
    return Ty;
  }
  case ExprKind::UN_OP: {
    auto *E = static_cast<UnaryExpr *>(&Exp);
    auto Ty = typeUnaryOperator(Ctx, E->Operator, *(E->Right));
    Exp.Ty = Ty;
    return Ty;
  }
  case ExprKind::CALL: {
    auto *E = static_cast<CallExpr *>(&Exp);
    auto T = typeInfer(Ctx, *(std::make_unique<VarExpr>(Exp.Loc, E->FuncName)));
    if (T->Tag != TypeTag::FUNC) {
      throw TypeError(Exp.Loc,
                      "Cannot call expression of type " + T->toString());
    }

    auto *FuncTy = static_cast<FuncT *>(T.get());
    auto ParamsSize = FuncTy->Params.size();
    auto ArgsSize = E->Args.size();
    if (ParamsSize != ArgsSize) {
      throw TypeError(Exp.Loc, "Expected " + std::to_string(ParamsSize) +
                                   " parameters, but got " +
                                   std::to_string(ArgsSize) + " arguments");
    }

    for (size_t Idx = 0; Idx < ParamsSize; Idx++) {
      typeCheck(Ctx, *(E->Args[Idx]),
                std::make_shared<Type>(FuncTy->Params[Idx]));
    }

    Exp.Ty = FuncTy->Ret;
    return FuncTy->Ret;
  }
  case ExprKind::FUNC:
    throw TypeError(Exp.Loc, "Unimplemented");
  }
}