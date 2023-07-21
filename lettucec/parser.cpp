#include "parser.hpp"
#include "AST.fwd.hpp"
#include "lexer.hpp"
#include <cstddef>
#include <memory>
// TODO: EOI?

Parser::Error::Error(Token Found)
    : std::runtime_error(Found.Filename + ":" + std::to_string(Found.Line) +
                         ":" + std::to_string(Found.Column) + ": " +
                         "Parser Error" + ": " + Found.Value),
      Found(Found) {}

Parser::Parser(Lexer &Lex) : Lex(Lex) {}

auto Parser::parse() -> std::unique_ptr<Expr> { return expression(); }

auto Parser::expression() -> std::unique_ptr<Expr> {
  if (auto LetOpt = accept({TokenTag::LET})) {
    return letExpression(LetOpt.value());
  }
  if (auto IfOpt = accept({TokenTag::IF})) {
    return ifExpression(IfOpt.value());
  }
  return term();
}

auto Parser::letExpression(Token StartToken) -> std::unique_ptr<Expr> {
  const Location Loc = {StartToken.Filename, StartToken.Line,
                        StartToken.Column};
  auto Name = expect({TokenTag::IDENT}).Value;
  expect({TokenTag::EQUAL});
  auto EqExpr = expression();
  expect({TokenTag::IN});
  auto InExpr = expression();
  return std::make_unique<LetExpr>(Loc, Name, std::move(EqExpr),
                                   std::move(InExpr));
}

auto Parser::ifExpression(Token StartToken) -> std::unique_ptr<Expr> {
  const Location Loc = {StartToken.Filename, StartToken.Line,
                        StartToken.Column};
  auto Condition = expression();
  expect({TokenTag::THEN});
  auto EqExpr = expression();
  expect({TokenTag::ELSE});
  auto InExpr = expression();
  return std::make_unique<IfExpr>(Loc, std::move(Condition), std::move(EqExpr),
                                  std::move(InExpr));
}

auto Parser::term() -> std::unique_ptr<Expr> {
  auto LeftExpr = factor();
  while (auto V = accept({TokenTag::PLUS, TokenTag::MINUS})) {
    const Location Loc = {V->Filename, V->Line, V->Column};
    auto Op = V->Tag;
    auto RightExpr = factor();
    LeftExpr = std::make_unique<BinaryExpr>(
        Loc, std::move(LeftExpr), TokenOp::TagToOp(Op), std::move(RightExpr));
  }
  return LeftExpr;
}

auto Parser::factor() -> std::unique_ptr<Expr> {
  auto LeftExpr = primary();
  while (auto V = accept({TokenTag::STAR, TokenTag::SLASH})) {
    const Location Loc = {V->Filename, V->Line, V->Column};
    auto Op = V->Tag;
    auto RightExpr = primary();
    LeftExpr = std::make_unique<BinaryExpr>(
        Loc, std::move(LeftExpr), TokenOp::TagToOp(Op), std::move(RightExpr));
  }
  return LeftExpr;
}

auto Parser::primary() -> std::unique_ptr<Expr> {
  if (accept({TokenTag::LPAREN})) {
    auto ParenExpr = expression();
    expect({TokenTag::RPAREN});
    return ParenExpr;
  }
  if (auto V = accept({TokenTag::INT})) {
    auto Value = std::stoi(V->Value);
    const Location Loc = {V->Filename, V->Line, V->Column};
    return std::make_unique<IntExpr>(Loc, Value);
  }
  if (auto V = accept({TokenTag::BOOL})) {
    auto Value = V->Value == "true";
    const Location Loc = {V->Filename, V->Line, V->Column};
    return std::make_unique<BoolExpr>(Loc, Value);
  }
  if (auto V = accept({TokenTag::IDENT})) {
    const Location Loc = {V->Filename, V->Line, V->Column};
    return std::make_unique<VarExpr>(Loc, V->Value);
  }
  throw Error(Lex.peek());
}

auto Parser::accept(std::initializer_list<TokenTag> Tags)
    -> std::optional<Token> {
  for (const auto &Tag : Tags) {
    if (check(Tag)) {
      return Lex.token();
    }
  }
  return std::nullopt;
}

//

auto Parser::expect(std::initializer_list<TokenTag> Tags) -> Token {
  auto Found = accept(Tags);
  if (!Found) {
    throw Error(Lex.peek());
  }
  return Found.value();
}

auto Parser::check(TokenTag Tag) -> bool {
  if (Lex.isDone()) {
    return false;
  }
  return Lex.peek().Tag == Tag;
}
