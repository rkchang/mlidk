#pragma once

#include "AST.hpp"
#include "lexer.hpp"
#include "types.hpp"
#include <memory>
#include <optional>

class Parser {
  class Error : public UserError {
  public:
    Error(Token Found)
        : UserError(Found.Filename, Found.Line, Found.Column,
                    "Parser Error: " + Found.Value) {}
  };

public:
  Parser(Lexer &Lex);
  auto parse() -> std::unique_ptr<RootNode>;

private:
  auto expression() -> std::unique_ptr<Expr>;
  auto logic() -> std::unique_ptr<Expr>;
  auto equality() -> std::unique_ptr<Expr>;
  auto comparisson() -> std::unique_ptr<Expr>;
  auto term() -> std::unique_ptr<Expr>;
  auto factor() -> std::unique_ptr<Expr>;
  auto unary() -> std::unique_ptr<Expr>;
  auto funcCall() -> std::unique_ptr<Expr>;
  auto primary() -> std::unique_ptr<Expr>;

  auto type() -> std::shared_ptr<Type>;

  // Accepts a consecutive sequence of tokens
  auto chainAccept(std::initializer_list<TokenTag> Tags)
      -> std::optional<std::vector<Token>>;
  // Accepts a token from a list
  auto accept(std::initializer_list<TokenTag> Tags) -> std::optional<Token>;
  // Expects a token from a list. Throws if not found!
  auto expect(std::initializer_list<TokenTag> Tags) -> Token;
  auto check(std::initializer_list<TokenTag> Tags) -> bool;

  // Helpers
  auto letExpression(Token StartToken) -> std::unique_ptr<Expr>;
  auto ifExpression(Token StartToken) -> std::unique_ptr<Expr>;
  auto funcLit(Token StartToken) -> std::unique_ptr<Expr>;

  auto identifier() -> std::string;

  Lexer &Lex;
};
