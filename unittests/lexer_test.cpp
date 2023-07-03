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