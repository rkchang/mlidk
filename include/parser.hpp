#pragma once

#include "AST.hpp"
#include "lexer.hpp"
#include <memory>
#include <optional>

class Parser {
  class Error : public std::runtime_error {
  public:
    const Token Found;

    Error(Token Found);
  };

public:
  Parser(Lexer &Lex);
  auto parse() -> std::unique_ptr<Expr>;

private:
  auto expression() -> std::unique_ptr<Expr>;
  auto logic() -> std::unique_ptr<Expr>;
  auto equality() -> std::unique_ptr<Expr>;
  auto comparisson() -> std::unique_ptr<Expr>;
  auto term() -> std::unique_ptr<Expr>;
  auto factor() -> std::unique_ptr<Expr>;
  auto unary() -> std::unique_ptr<Expr>;
  auto primary() -> std::unique_ptr<Expr>;
  // Accepts a token from a list
  auto accept(std::initializer_list<TokenTag> Tags) -> std::optional<Token>;
  // Expects a token from a list. Throws if not found!
  auto expect(std::initializer_list<TokenTag> Tags) -> Token;
  auto check(TokenTag Tag) -> bool;

  // Helpers
  auto letExpression(Token StartToken) -> std::unique_ptr<Expr>;
  auto ifExpression(Token StartToken) -> std::unique_ptr<Expr>;

  Lexer &Lex;
};
