#pragma once

#include "ASTVisitor.hpp"
#include "AST.hpp"
#include "lexer.hpp"
#include "types.hpp"

#include <any>
#include <memory>
#include <unordered_map>

using TypeCtx = std::unordered_map<std::string, std::shared_ptr<Type>>;

class TypeError : public UserError {
public:
  TypeError(Location Loc, std::string Message)
      : UserError(Loc.Filename, Loc.Line, Loc.Column,
                  "Type Error: " + Message){};
};

class TypeChecker : public ASTVisitor {
public:
  TypeCtx Ctx;
  TypeChecker(TypeCtx Ctx) : Ctx(Ctx) {}

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
};
