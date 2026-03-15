#pragma once

#include "AST.hpp"
#include "ASTVisitor.hpp"

#include <llvm/ADT/ScopedHashTable.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/SymbolTable.h>
#include <vector>

class MLIRGen : public ASTVisitor {
public:
  class Error : public std::runtime_error {
  public:
    const Location Loc;

    Error(Location Loc, std::string Msg);
  };

  MLIRGen(mlir::MLIRContext &Context);

  auto visit(RootNode &Node, std::any Context) -> std::any override;

  auto visit(DefExpr &Node, std::any Context) -> std::any override;
  auto visit(LetExpr &Node, std::any Context) -> std::any override;
  auto visit(IfExpr &Node, std::any Context) -> std::any override;
  auto visit(BinaryExpr &Node, std::any Context) -> std::any override;
  auto visit(UnaryExpr &Node, std::any Context) -> std::any override;
  auto visit(IntExpr &Node, std::any Context) -> std::any override;
  auto visit(BoolExpr &Node, std::any Context) -> std::any override;
  auto visit(VarExpr &Node, std::any Context) -> std::any override;
  auto visit(CallExpr &Node, std::any Context) -> std::any override;
  auto visit(FuncExpr &Node, std::any Context) -> std::any override;

  auto loc(const Location &Loc) -> mlir::Location;

  mlir::ModuleOp Module;

private:
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> SymbolTable;
  mlir::OpBuilder Buildr;
  int NextId = 0;
  auto freshName(std::string Prefix) -> std::string;

  auto generateFunction(Location Loca, std::any Context, std::string Name,
                        mlir::FunctionType FuncTy,
                        std::vector<std::pair<std::string, Type>> Params,
                        Expr &Body,
                        std::vector<std::pair<std::string, mlir::FunctionType>>
                            OtherDefinitions = {}) -> void;
  auto getOrCreateGlobalString(mlir::Location Loc, llvm::StringRef Name,
                               mlir::StringRef Value) -> mlir::Value;
  auto addArgs(const std::vector<std::unique_ptr<Expr>> &Args, std::any Context,
               std::vector<mlir::Value> &ArgsOut) -> void;
};
