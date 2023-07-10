#include "parser.hpp"
// TODO: EOI?

Parser::Error::Error(Token Found)
    : std::runtime_error(Found.Filename + ":" + std::to_string(Found.Line) +
                         ":" + std::to_string(Found.Column) + ": " +
                         "Parser Error" + ": " + Found.Value),
      Found(Found) {}

Parser::Parser(Lexer &Lex) : Lex(Lex) {}

auto Parser::parse() -> std::unique_ptr<Expr> { return expression(); }

auto Parser::expression() -> std::unique_ptr<Expr> {
  if (match({TokenTag::LET})) {
    auto V = match({TokenTag::IDENT});
    if (!V) {
      throw Error(Lex.peek());
    }
    auto Name = V->Value;
    if (!match({TokenTag::EQUAL})) {
      throw Error(Lex.peek());
    }
    auto EqExpr = expression();
    if (!match({TokenTag::IN})) {
      throw Error(Lex.peek());
    }
    auto InExpr = expression();
    return std::make_unique<LetExpr>(Name, std::move(EqExpr),
                                     std::move(InExpr));
  }
  return term();
}

auto Parser::term() -> std::unique_ptr<Expr> {
  auto LeftExpr = factor();
  while (auto V = match({TokenTag::PLUS, TokenTag::MINUS})) {
    auto Op = V->Tag;
    auto RightExpr = factor();
    LeftExpr = std::make_unique<BinaryExpr>(
        std::move(LeftExpr), TokenOp::TagToOp(Op), std::move(RightExpr));
  }
  return LeftExpr;
}

auto Parser::factor() -> std::unique_ptr<Expr> {
  auto LeftExpr = primary();
  while (auto V = match({TokenTag::STAR, TokenTag::SLASH})) {
    auto Op = V->Tag;
    auto RightExpr = primary();
    LeftExpr = std::make_unique<BinaryExpr>(
        std::move(LeftExpr), TokenOp::TagToOp(Op), std::move(RightExpr));
  }
  return LeftExpr;
}

auto Parser::primary() -> std::unique_ptr<Expr> {
  if (match({TokenTag::LPAREN})) {
    auto ParenExpr = expression();
    if (!match({TokenTag::RPAREN})) {
      throw Error(Lex.peek());
    }
    return ParenExpr;
  }
  if (auto V = match({TokenTag::INT})) {
    auto Num = std::stoi(V->Value);
    return std::make_unique<IntExpr>(Num);
  }
  if (auto V = match({TokenTag::IDENT})) {
    return std::make_unique<VarExpr>(V->Value);
  }
  throw Error(Lex.peek());
}

auto Parser::match(std::initializer_list<TokenTag> Tags)
    -> std::optional<Token> {
  for (const auto &Tag : Tags) {
    if (check(Tag)) {
      return Lex.token();
    }
  }
  return std::nullopt;
}

auto Parser::check(TokenTag Tag) -> bool {
  if (Lex.isDone()) {
    return false;
  }
  return Lex.peek().Tag == Tag;
}
