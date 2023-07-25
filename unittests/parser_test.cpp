#include "AST.fwd.hpp"
#include "AST.hpp"
#include "ASTPrinter.hpp"
#include "parser.hpp"
#include <deque>

#include <gtest/gtest.h>
#include <vector>

auto genAST(std::string Source) -> std::unique_ptr<Expr> {
  auto Lexr = Lexer(Source, "GTEST");
  auto Parsr = Parser(Lexr);
  auto AST = Parsr.parse();
  return std::move(AST->Exp);
}

TEST(ParserTest, LetExpr) {
  std::string Source = "let x = 1 in x";
  auto AST = genAST(Source);
  auto *Let = dynamic_cast<LetExpr *>(AST.get());
  EXPECT_NE(Let, nullptr) << "Expected LetExpr";

  auto *Int = dynamic_cast<IntExpr *>(Let->Value.get());
  EXPECT_NE(Int, nullptr) << "Expected IntExpr";

  auto *Body = dynamic_cast<VarExpr *>(Let->Body.get());
  EXPECT_NE(Body, nullptr) << "Expected VarExpr";
}

TEST(ParserTest, IfExpr) {
  std::string Source = "if true then x else 1";
  auto AST = genAST(Source);
  auto *If = dynamic_cast<IfExpr *>(AST.get());
  EXPECT_NE(If, nullptr) << "Expected IfExpr";

  auto *Cond = dynamic_cast<BoolExpr *>(If->Condition.get());
  EXPECT_NE(Cond, nullptr) << "Expected BoolExpr";

  auto *TrueBranch = dynamic_cast<VarExpr *>(If->TrueBranch.get());
  EXPECT_NE(TrueBranch, nullptr) << "Expected VarExpr";

  auto *FalseBranch = dynamic_cast<IntExpr *>(If->FalseBranch.get());
  EXPECT_NE(FalseBranch, nullptr) << "Expected IntExpr";
}

TEST(ParserTest, Logic) {
  std::vector<std::string> Ops = {"and", "or"};

  for (auto &Op : Ops) {
    std::string Source = "a == 1 " + Op + " true";
    auto AST = genAST(Source);
    auto *Cmp = dynamic_cast<BinaryExpr *>(AST.get());
    EXPECT_NE(Cmp, nullptr) << "Expected BinaryExpr";

    auto *Plus = dynamic_cast<BinaryExpr *>(Cmp->Left.get());
    EXPECT_NE(Plus, nullptr) << "Expected BinaryExpr";

    auto *A = dynamic_cast<VarExpr *>(Plus->Left.get());
    EXPECT_NE(A, nullptr) << "Expected VarExpr";

    auto *One = dynamic_cast<IntExpr *>(Plus->Right.get());
    EXPECT_NE(One, nullptr) << "Expected IntExpr";

    auto *True = dynamic_cast<BoolExpr *>(Cmp->Right.get());
    EXPECT_NE(True, nullptr) << "Expected BoolExpr";
  }
}

TEST(ParserTest, Equality) {
  std::vector<std::string> Ops = {"==", "!="};

  for (auto &Op : Ops) {
    std::string Source = "a < 1 " + Op + " true";
    auto AST = genAST(Source);
    auto *Cmp = dynamic_cast<BinaryExpr *>(AST.get());
    EXPECT_NE(Cmp, nullptr) << "Expected BinaryExpr";

    auto *Plus = dynamic_cast<BinaryExpr *>(Cmp->Left.get());
    EXPECT_NE(Plus, nullptr) << "Expected BinaryExpr";

    auto *A = dynamic_cast<VarExpr *>(Plus->Left.get());
    EXPECT_NE(A, nullptr) << "Expected VarExpr";

    auto *One = dynamic_cast<IntExpr *>(Plus->Right.get());
    EXPECT_NE(One, nullptr) << "Expected IntExpr";

    auto *True = dynamic_cast<BoolExpr *>(Cmp->Right.get());
    EXPECT_NE(True, nullptr) << "Expected BoolExpr";
  }
}

TEST(ParserTest, Comparisson) {
  std::vector<std::string> Ops = {"<", "<=", ">", ">="};

  for (auto &Op : Ops) {
    std::string Source = "a + 1 " + Op + " 2";
    auto AST = genAST(Source);
    auto *Cmp = dynamic_cast<BinaryExpr *>(AST.get());
    EXPECT_NE(Cmp, nullptr) << "Expected BinaryExpr";

    auto *Plus = dynamic_cast<BinaryExpr *>(Cmp->Left.get());
    EXPECT_NE(Plus, nullptr) << "Expected BinaryExpr";

    auto *A = dynamic_cast<VarExpr *>(Plus->Left.get());
    EXPECT_NE(A, nullptr) << "Expected VarExpr";

    auto *One = dynamic_cast<IntExpr *>(Plus->Right.get());
    EXPECT_NE(One, nullptr) << "Expected IntExpr";

    auto *Two = dynamic_cast<IntExpr *>(Cmp->Right.get());
    EXPECT_NE(Two, nullptr) << "Expected IntExpr";
  }
}

TEST(ParserTest, Arithmetic) {
  std::string Source = "(1 + a) * b";
  auto AST = genAST(Source);
  auto *Factor = dynamic_cast<BinaryExpr *>(AST.get());
  EXPECT_NE(Factor, nullptr) << "Expected BinaryExpr";

  auto *Term = dynamic_cast<BinaryExpr *>(Factor->Left.get());
  EXPECT_NE(Term, nullptr) << "Expected BinaryExpr";

  auto *One = dynamic_cast<IntExpr *>(Term->Left.get());
  EXPECT_NE(One, nullptr) << "Expected IntExpr";

  auto *A = dynamic_cast<VarExpr *>(Term->Right.get());
  EXPECT_NE(A, nullptr) << "Expected VarExpr";

  auto *Var = dynamic_cast<VarExpr *>(Factor->Right.get());
  EXPECT_NE(Var, nullptr) << "Expected VarExpr";
}

TEST(ParserTest, UnaryNot) {
  std::string Source = "not true";
  auto AST = genAST(Source);
  auto *Not = dynamic_cast<UnaryExpr *>(AST.get());
  EXPECT_NE(Not, nullptr) << "Expected UnaryExpr";

  auto *True = dynamic_cast<BoolExpr *>(Not->Right.get());
  EXPECT_NE(True, nullptr) << "Expected BoolExpr";
}

TEST(ParserTest, NestedLetExpr) {
  std::string Source = "let x = 1 in let y = 2 in x + y";
  auto AST = genAST(Source);
  auto *Let = dynamic_cast<LetExpr *>(AST.get());
  EXPECT_NE(Let, nullptr) << "Expected LetExpr";

  auto *LetValue = dynamic_cast<IntExpr *>(Let->Value.get());
  EXPECT_NE(LetValue, nullptr) << "Expected IntExpr";

  auto *Let2 = dynamic_cast<LetExpr *>(Let->Body.get());
  EXPECT_NE(Let2, nullptr) << "Expected LetExpr";

  auto *Let2Value = dynamic_cast<IntExpr *>(Let2->Value.get());
  EXPECT_NE(Let2Value, nullptr) << "Expected IntExpr";

  auto *Let2Body = dynamic_cast<BinaryExpr *>(Let2->Body.get());
  EXPECT_NE(Let2Body, nullptr) << "Expected BinaryExpr";

  auto *X = dynamic_cast<VarExpr *>(Let2Body->Right.get());
  EXPECT_NE(X, nullptr) << "Expected VarExpr";

  auto *Y = dynamic_cast<VarExpr *>(Let2Body->Right.get());
  EXPECT_NE(Y, nullptr) << "Expected VarExpr";
}

TEST(ParserTest, CallExpr) {
  std::string Source = "let x = f(1+2, a) in x";
  auto AST = genAST(Source);
  auto *Let = dynamic_cast<LetExpr *>(AST.get());
  EXPECT_NE(Let, nullptr) << "Expected LetExpr";

  auto *LetValue = dynamic_cast<CallExpr *>(Let->Value.get());
  EXPECT_NE(LetValue, nullptr) << "Expected CallExpr";

  auto *CallExprArg0 = dynamic_cast<BinaryExpr *>(LetValue->Args[0].get());
  EXPECT_NE(CallExprArg0, nullptr) << "Expected BinExpr";

  auto *CallExprArg1 = dynamic_cast<VarExpr *>(LetValue->Args[1].get());
  EXPECT_NE(CallExprArg1, nullptr) << "Expected VarExpr";

  auto *One = dynamic_cast<IntExpr *>(CallExprArg0->Left.get());
  EXPECT_NE(One, nullptr) << "Expected IntExpr";

  auto *Two = dynamic_cast<IntExpr *>(CallExprArg0->Right.get());
  EXPECT_NE(Two, nullptr) << "Expected IntExpr";
}

TEST(ParserTest, FuncExprNoParams) {
  std::string Source = "|| 1";
  auto AST = genAST(Source);
  auto *Func = dynamic_cast<FuncExpr *>(AST.get());
  EXPECT_NE(Func, nullptr) << "Expected FuncExpr";

  auto Params = Func->Params;
  EXPECT_EQ(Params.size(), 0) << "Expected 0 Parameters";

  auto *Body = dynamic_cast<IntExpr *>(Func->Body.get());
  EXPECT_NE(Body, nullptr) << "Expected IntExpr";
}

TEST(ParserTest, FuncExprOneParam) {
  std::string Source = "|x: i32| 1";
  auto AST = genAST(Source);
  auto *Func = dynamic_cast<FuncExpr *>(AST.get());
  EXPECT_NE(Func, nullptr) << "Expected FuncExpr";

  auto Params = Func->Params;
  EXPECT_EQ(Params.size(), 1) << "Expected 1 Parameter";

  auto *Body = dynamic_cast<IntExpr *>(Func->Body.get());
  EXPECT_NE(Body, nullptr) << "Expected IntExpr";
}

TEST(ParserTest, FuncExprTwoParams) {
  std::string Source = "|x: i32, y: i32| 1";
  auto AST = genAST(Source);
  auto *Func = dynamic_cast<FuncExpr *>(AST.get());
  EXPECT_NE(Func, nullptr) << "Expected FuncExpr";

  auto Params = Func->Params;
  EXPECT_EQ(Params.size(), 2) << "Expected 2 Parameters";

  auto *Body = dynamic_cast<IntExpr *>(Func->Body.get());
  EXPECT_NE(Body, nullptr) << "Expected IntExpr";
}