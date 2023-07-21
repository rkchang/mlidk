#include "AST.fwd.hpp"
#include "ASTPrinter.hpp"
#include "parser.hpp"
#include <deque>

#include <gtest/gtest.h>

auto genAST(std::string Source) -> std::unique_ptr<Expr> {
  auto Lexr = Lexer(Source, "GTEST");
  auto Parsr = Parser(Lexr);
  auto AST = Parsr.parse();
  return AST;
}

TEST(ParserTest, LetExpr) {
  std::string Source = "let x = 1 in x";
  auto AST = genAST(Source);
  auto Let = dynamic_cast<LetExpr *>(AST.get());
  EXPECT_NE(Let, nullptr) << "Expected LetExpr";

  auto Int = dynamic_cast<IntExpr *>(Let->Value.get());
  EXPECT_NE(Int, nullptr) << "Expected IntExpr";

  auto Body = dynamic_cast<VarExpr *>(Let->Body.get());
  EXPECT_NE(Body, nullptr) << "Expected VarExpr";
}

TEST(ParserTest, IfExpr) {
  std::string Source = "if true then x else 1";
  auto AST = genAST(Source);
  auto If = dynamic_cast<IfExpr *>(AST.get());
  EXPECT_NE(If, nullptr) << "Expected IfExpr";

  auto Cond = dynamic_cast<BoolExpr *>(If->Condition.get());
  EXPECT_NE(Cond, nullptr) << "Expected BoolExpr";

  auto TrueBranch = dynamic_cast<VarExpr *>(If->TrueBranch.get());
  EXPECT_NE(TrueBranch, nullptr) << "Expected VarExpr";

  auto FalseBranch = dynamic_cast<IntExpr *>(If->FalseBranch.get());
  EXPECT_NE(FalseBranch, nullptr) << "Expected IntExpr";
}

TEST(ParserTest, Arithmetic) {
  std::string Source = "(1 + a) * b";
  auto AST = genAST(Source);
  auto Factor = dynamic_cast<BinaryExpr *>(AST.get());
  EXPECT_NE(Factor, nullptr) << "Expected BinaryExpr";

  auto Term = dynamic_cast<BinaryExpr *>(Factor->Left.get());
  EXPECT_NE(Term, nullptr) << "Expected BinaryExpr";

  auto One = dynamic_cast<IntExpr *>(Term->Left.get());
  EXPECT_NE(One, nullptr) << "Expected IntExpr";

  auto a = dynamic_cast<VarExpr *>(Term->Right.get());
  EXPECT_NE(a, nullptr) << "Expected VarExpr";

  auto Var = dynamic_cast<VarExpr *>(Factor->Right.get());
  EXPECT_NE(Var, nullptr) << "Expected VarExpr";
}

TEST(ParserTest, NestedLetExpr) {
  std::string Source = "let x = 1 in let y = 2 in x + y";
  auto AST = genAST(Source);
  auto Let = dynamic_cast<LetExpr *>(AST.get());
  EXPECT_NE(Let, nullptr) << "Expected LetExpr";

  auto LetValue = dynamic_cast<IntExpr *>(Let->Value.get());
  EXPECT_NE(LetValue, nullptr) << "Expected IntExpr";

  auto Let2 = dynamic_cast<LetExpr *>(Let->Body.get());
  EXPECT_NE(Let2, nullptr) << "Expected LetExpr";

  auto Let2Value = dynamic_cast<IntExpr *>(Let2->Value.get());
  EXPECT_NE(Let2Value, nullptr) << "Expected IntExpr";

  auto Let2Body = dynamic_cast<BinaryExpr *>(Let2->Body.get());
  EXPECT_NE(Let2Body, nullptr) << "Expected BinaryExpr";

  auto x = dynamic_cast<VarExpr *>(Let2Body->Right.get());
  EXPECT_NE(x, nullptr) << "Expected VarExpr";

  auto y = dynamic_cast<VarExpr *>(Let2Body->Right.get());
  EXPECT_NE(y, nullptr) << "Expected VarExpr";
}
