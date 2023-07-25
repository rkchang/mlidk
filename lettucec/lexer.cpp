#include <lexer.hpp>

#include <optional>
#include <string>
#include <unordered_map>

auto TokenOp::TagToOp(TokenTag Tag) -> OpType {
  std::unordered_map<TokenTag, OpType> Map{
      {TokenTag::PLUS, OpType::ADD},
      {TokenTag::MINUS, OpType::MINUS},
      {TokenTag::STAR, OpType::MUL},
      {TokenTag::SLASH, OpType::DIV},
      {TokenTag::EQUAL_EQUAL, OpType::EQ},
      {TokenTag::BANG_EQUAL, OpType::NE},
      {TokenTag::LESS, OpType::LT},
      {TokenTag::LESS_EQUAL, OpType::LE},
      {TokenTag::GREATER, OpType::GT},
      {TokenTag::GREATER_EQUAL, OpType::GE},
      {TokenTag::AND, OpType::AND},
      {TokenTag::OR, OpType::OR},
      {TokenTag::NOT, OpType::NOT},
  };
  return Map[Tag];
}

auto TokenOp::OpToStr(OpType Op) -> std::string {
  std::unordered_map<OpType, std::string> Map{
      {OpType::ADD, "+"},   {OpType::MINUS, "-"}, {OpType::MUL, "*"},
      {OpType::DIV, "/"},   {OpType::EQ, "=="},   {OpType::NE, "!="},
      {OpType::LT, "<"},    {OpType::LE, "<="},   {OpType::GT, ">"},
      {OpType::GE, ">="},   {OpType::AND, "and"}, {OpType::OR, "or"},
      {OpType::NOT, "not"},
  };
  return Map[Op];
}

// All keywords in language
const std::unordered_map<std::string, TokenTag> KeyWords = {
    {"let", TokenTag::LET},      {"in", TokenTag::IN},
    {"if", TokenTag::IF},        {"then", TokenTag::THEN},
    {"else", TokenTag::ELSE},    {"true", TokenTag::BOOL},
    {"false", TokenTag::BOOL},   {"and", TokenTag::AND},
    {"or", TokenTag::OR},        {"not", TokenTag::NOT},
    {"fun", TokenTag::FUN},      {"i32", TokenTag::I32},
    {"bool", TokenTag::BOOL_KW}, {"void", TokenTag::VOID}};

//--------
// Lexer
//--------

Lexer::Lexer(std::string_view Source, std::string Filename)
    : State{0, 1, 1}, Size(Source.length()), Source(Source),
      Filename(Filename) {}

auto Lexer::token() -> Token {
  // Skip all leading whitespace
  takeWhile(std::isspace);
  if (isDone()) {
    return Token{TokenTag::EOI, "", Filename, State.Line, State.Column};
  }

  // Keep track of start position
  auto Char = Source[State.Index];
  auto StartLine = State.Line;
  auto StartCol = State.Column;
  switch (Char) {
  case '+':
    step();
    return Token{TokenTag::PLUS, "+", Filename, StartLine, StartCol};
  case '-': {
    step();
    auto V = lookahead(0);
    if (V.value_or('\0') == '>') {
      step();
      return Token{TokenTag::ARROW, "->", Filename, StartLine, StartCol};
    }
    return Token{TokenTag::MINUS, "-", Filename, StartLine, StartCol};
  }
  case '*':
    step();
    return Token{TokenTag::STAR, "*", Filename, StartLine, StartCol};
  case '/':
    step();
    return Token{TokenTag::SLASH, "/", Filename, StartLine, StartCol};
  case '=': {
    step();
    auto V = lookahead(0);
    if (V.value_or('\0') == '=') {
      step();
      return Token{TokenTag::EQUAL_EQUAL, "==", Filename, StartLine, StartCol};
    }
    return Token{TokenTag::EQUAL, "=", Filename, StartLine, StartCol};
  }
  case '<': {
    step();
    auto V = lookahead(0);
    if (V.value_or('\0') == '=') {
      step();
      return Token{TokenTag::LESS_EQUAL, "<=", Filename, StartLine, StartCol};
    }
    return Token{TokenTag::LESS, "<", Filename, StartLine, StartCol};
  }
  case '>': {
    step();
    auto V = lookahead(0);
    if (V.value_or('\0') == '=') {
      step();
      return Token{TokenTag::GREATER_EQUAL, ">=", Filename, StartLine,
                   StartCol};
    }
    return Token{TokenTag::GREATER, ">", Filename, StartLine, StartCol};
  }
  case '!': {
    // peek past '!' to next character
    auto V = lookahead(1);
    if (V.value_or('\0') == '=') {
      step();
      step();
      return Token{TokenTag::BANG_EQUAL, "!=", Filename, StartLine, StartCol};
    }
    // '!' is not a recognized token
    throw Error(State.Line, State.Column, "Invalid Token", Filename);
  }
  case '(':
    step();
    return Token{TokenTag::LPAREN, "(", Filename, StartLine, StartCol};
  case ')':
    step();
    return Token{TokenTag::RPAREN, ")", Filename, StartLine, StartCol};
  case ',':
    step();
    return Token{TokenTag::COMMA, ",", Filename, StartLine, StartCol};

  default:
    if (std::isdigit(Char)) {
      // Int
      auto Value = takeWhile(std::isdigit);
      return Token{TokenTag::INT, Value, Filename, StartLine, StartCol};

    } else if (std::isalpha(Char)) {
      // Ident
      auto Value =
          takeWhile([](char C) { return std::isdigit(C) || std::isalpha(C); });
      auto Tag =
          KeyWords.contains(Value) ? KeyWords.at(Value) : TokenTag::IDENT;
      return Token{Tag, Value, Filename, StartLine, StartCol};
    }
  }
  throw Error(State.Line, State.Column, "Invalid Token", Filename);
}

auto Lexer::peek() -> Token {
  // This function should be const,
  // but can't be because token() isn't.
  auto OldState = State;
  auto Token = token();
  State = OldState;
  return Token;
}

auto Lexer::getState() -> SrcLoc const { return State; }

auto Lexer::setState(Lexer::SrcLoc NewState) -> void { State = NewState; }

auto Lexer::isDone() -> bool { return State.Index >= Size; }

auto Lexer::lookahead(int N) -> std::optional<char> {
  if (State.Index + N >= Size) {
    return std::nullopt;
  }
  return std::make_optional(Source[State.Index + N]);
}

auto Lexer::step() -> char {
  // TODO: Check for out of range
  auto Char = Source[State.Index];
  State.Index += 1;
  if (Char == '\n') {
    State.Line += 1;
    State.Column = 1;
  } else {
    State.Column += 1;
  }
  return Char;
}

auto Lexer::takeWhile(std::function<bool(char)> Predicate) -> std::string {
  auto Start = State.Index;
  while (!isDone() && Predicate(Source[State.Index])) {
    step();
  }
  std::string SubStr{Source.substr(Start, State.Index - Start)};
  return SubStr;
}