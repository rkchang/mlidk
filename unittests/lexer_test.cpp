#include "lexer.hpp"

#include <gtest/gtest.h>

TEST(LexerTest, Example) {
  Lexer Lexer("hello world", "hello.cpp");
  Token Tok = Lexer.lex();
  EXPECT_EQ(Tok.Value, "hello");
}