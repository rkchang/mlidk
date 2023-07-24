#include "MLIR_gen.hpp"
#include "AST.fwd.hpp"
#include "lexer.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>

#include <any>
#include <cstddef>

#include <iostream>

MLIRGen::Error::Error(Location Loc, std::string Msg)
    : std::runtime_error(Loc.Filename + ":" + std::to_string(Loc.Line) + ":" +
                         std::to_string(Loc.Column) + ": " +
                         "MLIR Generator Error" + ": " + Msg),
      Loc(Loc) {}

MLIRGen::MLIRGen(mlir::MLIRContext &Context) : Buildr(&Context) {
  Module = mlir::ModuleOp::create(Buildr.getUnknownLoc());
  Buildr.setInsertionPointToEnd(Module.getBody());
}

auto MLIRGen::loc(const Location &Loc) -> mlir::Location {
  return mlir::FileLineColLoc::get(Buildr.getStringAttr(Loc.Filename), Loc.Line,
                                   Loc.Column);
}

//

auto MLIRGen::visit(const RootNode &Node, std::any Context) -> std::any {
  auto Loc = loc(Node.Loc);
  auto Ty = Buildr.getFunctionType(std::nullopt, {Buildr.getI32Type()});
  auto Fun = Buildr.create<mlir::func::FuncOp>(Loc, "_mlir_ciface_main", Ty);
  Fun.addEntryBlock();
  auto &Blk = Fun.front();
  Buildr.setInsertionPointToStart(&Blk);

  auto V = Node.Exp->accept(*this, Context);

  auto Value = std::any_cast<mlir::Value>(V);
  Buildr.create<mlir::func::ReturnOp>(Loc, Value);

  // TODO: Return type must change depending on whether int or bool expr!
  auto Ty2 = Buildr.getFunctionType(std::nullopt, {Value.getType()});
  Fun.setType(Ty2);

  return Fun;
}

auto MLIRGen::visit(const LetExpr &Node, std::any Context) -> std::any {
  const llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> Scope(
      SymbolTable);
  auto V = Node.Value->accept(*this, Context);
  auto Value = std::any_cast<mlir::Value>(V);
  SymbolTable.insert(Node.Name, Value);
  return Node.Body->accept(*this, Context);
}

auto MLIRGen::visit(const IfExpr &Node, std::any Context) -> std::any {
  // TODO: Check type?
  auto TR = mlir::TypeRange(Buildr.getI1Type());

  auto Cond =
      std::any_cast<mlir::Value>(Node.Condition->accept(*this, Context));
  auto If = Buildr.create<mlir::scf::IfOp>(loc(Node.Loc), TR, Cond, true);

  auto OldBuildr = Buildr;

  // Then branch
  auto *Then = &If.getThenRegion();
  Buildr = mlir::OpBuilder(Then);

  auto TrueValue =
      std::any_cast<mlir::Value>(Node.TrueBranch->accept(*this, Context));
  auto TrueYield =
      Buildr.create<mlir::scf::YieldOp>(loc(Node.TrueBranch->Loc), TrueValue);

  // Else branch
  auto *Else = &If.getElseRegion();
  Buildr = mlir::OpBuilder(Else);
  auto FalseValue =
      std::any_cast<mlir::Value>(Node.FalseBranch->accept(*this, Context));
  auto FalseYield =
      Buildr.create<mlir::scf::YieldOp>(loc(Node.FalseBranch->Loc), FalseValue);

  Buildr = OldBuildr;

  auto Result = If->getOpResult(0);

  return static_cast<mlir::Value>(Result);
}

auto MLIRGen::visit(const BinaryExpr &Node, std::any Context) -> std::any {
  auto Lhs = std::any_cast<mlir::Value>(Node.Left->accept(*this, Context));
  auto Rhs = std::any_cast<mlir::Value>(Node.Right->accept(*this, Context));
  auto DataType = Buildr.getI32Type(); // TODO: get type from lhs?
  switch (Node.Operator) {
  case TokenOp::OpType::ADD:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::AddIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::MUL:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::MulIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::MINUS:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::SubIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::DIV:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::DivSIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::EQ:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::eq, Lhs, Rhs));
  case TokenOp::OpType::NE:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::ne, Lhs, Rhs));
  case TokenOp::OpType::LT:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::slt, Lhs, Rhs));
  case TokenOp::OpType::LE:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::sle, Lhs, Rhs));
  case TokenOp::OpType::GT:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::sgt, Lhs, Rhs));
  case TokenOp::OpType::GE:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::sge, Lhs, Rhs));
  case TokenOp::OpType::AND:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::AndIOp>(loc(Node.Loc), Lhs, Rhs));
  case TokenOp::OpType::OR:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::OrIOp>(loc(Node.Loc), Lhs, Rhs));

  case TokenOp::OpType::NOT:
    throw Error(Node.Loc, "Unsupported operation");
  }
  throw Error(Node.Loc, "Unknown binary operator");
}

auto MLIRGen::visit(const UnaryExpr &Node, std::any Context) -> std::any {
  switch (Node.Operator) {
  case TokenOp::OpType::NOT: {
    auto Rhs = std::any_cast<mlir::Value>(Node.Right->accept(*this, Context));
    // TODO: Fix this mess
    auto DataType = Buildr.getI1Type();
    auto TrueAttr = Buildr.getBoolAttr(true);
    auto FalseAttr = Buildr.getBoolAttr(false);
    auto True = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                       TrueAttr);
    auto False = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                        FalseAttr);

    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::SelectOp>(loc(Node.Loc), Rhs, False, True));
  }

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
    throw Error(Node.Loc, "Unsupported operation");
  }
}

auto MLIRGen::visit(const IntExpr &Node, std::any) -> std::any {
  auto DataType = Buildr.getI32Type();
  auto DataAttribute = Buildr.getI32IntegerAttr(Node.Value);
  auto Op = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                   DataAttribute);
  return static_cast<mlir::Value>(Op);
}

auto MLIRGen::visit(const BoolExpr &Node, std::any) -> std::any {
  auto DataType = Buildr.getI1Type();
  auto DataAttribute = Buildr.getBoolAttr(Node.Value);
  auto Op = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                   DataAttribute);
  return static_cast<mlir::Value>(Op);
}

auto MLIRGen::visit(const VarExpr &Node, std::any) -> std::any {
  if (auto Variable = SymbolTable.lookup(Node.Name)) {
    return Variable;
  }
  throw Error(Node.Loc, "Unknown variable reference: " + Node.Name);
}
