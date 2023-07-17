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
  auto term() -> std::unique_ptr<Expr>;
  auto factor() -> std::unique_ptr<Expr>;
  auto primary() -> std::unique_ptr<Expr>;
  auto match(std::initializer_list<TokenTag> Tags) -> std::optional<Token>;
  auto check(TokenTag Tag) -> bool;

  Lexer &Lex;
};
