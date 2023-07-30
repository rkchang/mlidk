#pragma once

// Forward declare classes to handle circular dependency between AST.hpp and
// ASTVisitor.hpp
class RootNode;
class DefExpr;
class LetExpr;
class IfExpr;
class BinaryExpr;
class UnaryExpr;
class IntExpr;
class BoolExpr;
class VarExpr;
class CallExpr;
class FuncExpr;
