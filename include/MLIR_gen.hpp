#pragma once

#include "AST.hpp"
#include "ASTVisitor.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ScopedHashTable.h"

class MLIRGen : public ASTVisitor {
public:
  class Error : public std::runtime_error {
  public:
    const Location Loc;

    Error(Location Loc, std::string Msg);
  };

  MLIRGen(mlir::MLIRContext &Context);
  auto visit(const LetExpr &Node, std::any Context) -> std::any override;
  auto visit(const IfExpr &Node, std::any Context) -> std::any override;
  auto visit(const BinaryExpr &Node, std::any Context) -> std::any override;
  auto visit(const UnaryExpr &Node, std::any Context) -> std::any override;
  auto visit(const IntExpr &Node, std::any Context) -> std::any override;
  auto visit(const BoolExpr &Node, std::any Context) -> std::any override;
  auto visit(const VarExpr &Node, std::any Context) -> std::any override;
  auto loc(const Location &Loc) -> mlir::Location;

  mlir::ModuleOp Module;

private:
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> SymbolTable;
  mlir::OpBuilder Buildr;
};
