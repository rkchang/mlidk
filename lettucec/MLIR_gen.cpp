#include "MLIR_gen.hpp"
#include "AST.fwd.hpp"
#include "lexer.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"

#include <any>
#include <cstddef>

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

auto MLIRGen::visit(const IfExpr &Node, std::any) -> std::any {
  throw Error(Node.Loc, "Unsupported operation");
}

auto MLIRGen::visit(const BinaryExpr &Node, std::any Context) -> std::any {
  auto Lhs = std::any_cast<mlir::Value>(Node.Left->accept(*this, Context));
  auto Rhs = std::any_cast<mlir::Value>(Node.Right->accept(*this, Context));
  auto DataType = Buildr.getI32Type(); // TODO: get type from lhs?
  switch (Node.Operator) {
  case TokenOp::OpType::ADD:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::AddIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
    break;
  case TokenOp::OpType::MUL:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::MulIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
    break;
  case TokenOp::OpType::MINUS:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::SubIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
    break;
  case TokenOp::OpType::DIV:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::DivSIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
    break;
  }
  throw Error(Node.Loc, "Unknown binary operator");
}

auto MLIRGen::visit(const UnaryExpr &Node, std::any) -> std::any {
  throw Error(Node.Loc, "Unsupported operation");
}

auto MLIRGen::visit(const IntExpr &Node, std::any) -> std::any {
  auto DataType = Buildr.getI32Type();
  auto DataAttribute = Buildr.getI32IntegerAttr(Node.Value);
  auto Op = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                   DataAttribute);
  return static_cast<mlir::Value>(Op);
}

auto MLIRGen::visit(const BoolExpr &Node, std::any) -> std::any {
  throw Error(Node.Loc, "Unsupported operation");
}

auto MLIRGen::visit(const VarExpr &Node, std::any) -> std::any {
  if (auto Variable = SymbolTable.lookup(Node.Name)) {
    return Variable;
  }
  throw Error(Node.Loc, "Unknown variable reference: " + Node.Name);
}
