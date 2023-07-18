#include "MLIR_gen.hpp"
#include "Romaine/RomaineOps.h"
#include "lexer.hpp"

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
auto MLIRGen::visit(const LetExpr &Node, std::any Context) -> std::any {
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> Scope(SymbolTable);
  auto V = Node.Value->accept(*this, Context);
  auto Value = std::any_cast<mlir::Value>(V);
  SymbolTable.insert(Node.Name, Value);
  Node.Body->accept(*this, Context);
  return NULL;
}
auto MLIRGen::visit(const BinaryExpr &Node, std::any Context) -> std::any {
  auto lhs = std::any_cast<mlir::Value>(Node.Left->accept(*this, Context));
  auto rhs = std::any_cast<mlir::Value>(Node.Right->accept(*this, Context));
  auto dataType = Buildr.getI32Type(); // TODO: get type from lhs?
  switch(Node.Operator) {
  case TokenOp::OpType::ADD:
    return static_cast<mlir::Value>(Buildr.create<mlir::romaine::AddOp>(loc(Node.Loc), dataType, lhs, rhs));
    break;
  case TokenOp::OpType::MUL:
    return static_cast<mlir::Value>(Buildr.create<mlir::romaine::MulOp>(loc(Node.Loc), dataType, lhs, rhs));
    break;
  }
  throw Error(Node.Loc, "Unknown binary operator");
}
auto MLIRGen::visit(const IntExpr &Node, std::any) -> std::any {
  auto dataType = Buildr.getI32Type();
  auto dataAttribute = Buildr.getI32IntegerAttr(Node.Value);
  auto op = Buildr.create<mlir::romaine::ConstantOp>(loc(Node.Loc), dataType, dataAttribute);
  return static_cast<mlir::Value>(op);
}
auto MLIRGen::visit(const VarExpr &Node, std::any) -> std::any {
  if (auto Variable = SymbolTable.lookup(Node.Name)) {
    return Variable;
  }
  throw Error(Node.Loc, "Unknown variable reference: " + Node.Name);
}
