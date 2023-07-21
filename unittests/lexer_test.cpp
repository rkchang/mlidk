#include "lexer.hpp"

#include <gtest/gtest.h>
#include <vector>

TEST(LexerTest, LexEOI) {
  Lexer Lexer("", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::EOI);
  EXPECT_EQ(Tok.Value, "");
}

TEST(LexerTest, LexPeek) {
  Lexer Lexer("123", "hello.cpp");
  Token Tok1 = Lexer.peek();
  EXPECT_EQ(Tok1.Tag, TokenTag::INT);
  EXPECT_EQ(Tok1.Value, "123");

  Token Tok2 = Lexer.peek();
  EXPECT_EQ(Tok2.Tag, TokenTag::INT);
  EXPECT_EQ(Tok2.Value, "123");
}

TEST(LexerTest, LexInt) {
  Lexer Lexer("123", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::INT);
  EXPECT_EQ(Tok.Value, "123");
}

TEST(LexerTest, LexTrue) {
  Lexer Lexer("true", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::BOOL);
  EXPECT_EQ(Tok.Value, "true");
}

TEST(LexerTest, LexFalse) {
  Lexer Lexer("false", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::BOOL);
  EXPECT_EQ(Tok.Value, "false");
}

TEST(LexerTest, LexIdent) {
  Lexer Lexer("abc", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::IDENT);
  EXPECT_EQ(Tok.Value, "abc");
}

TEST(LexerTest, LexSkipWhitespace) {
  Lexer Lexer("  \t \n abc", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::IDENT);
  EXPECT_EQ(Tok.Value, "abc");
}

TEST(LexerTest, LexLet) {
  Lexer Lexer("let", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::LET);
  EXPECT_EQ(Tok.Value, "let");
}

TEST(LexerTest, LexIn) {
  Lexer Lexer("in", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::IN);
  EXPECT_EQ(Tok.Value, "in");
}

TEST(LexerTest, LexIf) {
  Lexer Lexer("if", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::IF);
  EXPECT_EQ(Tok.Value, "if");
}

TEST(LexerTest, LexThen) {
  Lexer Lexer("then", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::THEN);
  EXPECT_EQ(Tok.Value, "then");
}

TEST(LexerTest, LexElse) {
  Lexer Lexer("else", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::ELSE);
  EXPECT_EQ(Tok.Value, "else");
}

TEST(LexerTest, LexPlus) {
  Lexer Lexer("+", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::PLUS);
  EXPECT_EQ(Tok.Value, "+");
}

TEST(LexerTest, LexMinus) {
  Lexer Lexer("-", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::MINUS);
  EXPECT_EQ(Tok.Value, "-");
}

TEST(LexerTest, LexStar) {
  Lexer Lexer("*", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::STAR);
  EXPECT_EQ(Tok.Value, "*");
}

TEST(LexerTest, LexSlash) {
  Lexer Lexer("/", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::SLASH);
  EXPECT_EQ(Tok.Value, "/");
}

TEST(LexerTest, LexEqualEqual) {
  Lexer Lexer("==", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::EQUAL_EQUAL);
  EXPECT_EQ(Tok.Value, "==");
}

TEST(LexerTest, LexBangEqual) {
  Lexer Lexer("!=", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::BANG_EQUAL);
  EXPECT_EQ(Tok.Value, "!=");
}

TEST(LexerTest, LexLess) {
  Lexer Lexer("<", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::LESS);
  EXPECT_EQ(Tok.Value, "<");
}

TEST(LexerTest, LexLessEqual) {
  Lexer Lexer("<=", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::LESS_EQUAL);
  EXPECT_EQ(Tok.Value, "<=");
}

TEST(LexerTest, LexGreater) {
  Lexer Lexer(">", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::GREATER);
  EXPECT_EQ(Tok.Value, ">");
}

TEST(LexerTest, LexGreaterEqual) {
  Lexer Lexer(">=", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::GREATER_EQUAL);
  EXPECT_EQ(Tok.Value, ">=");
}

TEST(LexerTest, LexAnd) {
  Lexer Lexer("and", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::AND);
  EXPECT_EQ(Tok.Value, "and");
}

TEST(LexerTest, LexOr) {
  Lexer Lexer("or", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::OR);
  EXPECT_EQ(Tok.Value, "or");
}

TEST(LexerTest, LexNot) {
  Lexer Lexer("not", "hello.cpp");
  Token Tok = Lexer.token();
  EXPECT_EQ(Tok.Tag, TokenTag::NOT);
  EXPECT_EQ(Tok.Value, "not");
}

TEST(LexerTest, LexMany) {
  Lexer Lexer("1 + x", "hello.cpp");
  std::vector<Token> Tokens;
  while (!Lexer.isDone()) {
    Tokens.push_back(Lexer.token());
  }
  EXPECT_EQ(Tokens[0].Tag, TokenTag::INT);
  EXPECT_EQ(Tokens[0].Value, "1");

  EXPECT_EQ(Tokens[1].Tag, TokenTag::PLUS);
  EXPECT_EQ(Tokens[1].Value, "+");

  EXPECT_EQ(Tokens[2].Tag, TokenTag::IDENT);
  EXPECT_EQ(Tokens[2].Value, "x");

  EXPECT_EQ(Tokens.size(), 3);
}

TEST(LexerTest, LexLetInExpr) {
  Lexer Lexer("let i = 1 + x", "hello.cpp");
  std::vector<Token> Tokens;
  while (!Lexer.isDone()) {
    Tokens.push_back(Lexer.token());
  }
  EXPECT_EQ(Tokens[0].Tag, TokenTag::LET);
  EXPECT_EQ(Tokens[0].Value, "let");

  EXPECT_EQ(Tokens[1].Tag, TokenTag::IDENT);
  EXPECT_EQ(Tokens[1].Value, "i");

  EXPECT_EQ(Tokens[2].Tag, TokenTag::EQUAL);
  EXPECT_EQ(Tokens[2].Value, "=");

  EXPECT_EQ(Tokens[3].Tag, TokenTag::INT);
  EXPECT_EQ(Tokens[3].Value, "1");

  EXPECT_EQ(Tokens[4].Tag, TokenTag::PLUS);
  EXPECT_EQ(Tokens[4].Value, "+");

  EXPECT_EQ(Tokens[5].Tag, TokenTag::IDENT);
  EXPECT_EQ(Tokens[5].Value, "x");

  EXPECT_EQ(Tokens.size(), 6);
}