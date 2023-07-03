#include <lexer.hpp>
#include <map>

// All keywords in language
const std::map<std::string, TokenTag> KeyWords = {{"let", TokenTag::LET},
                                                  {"in", TokenTag::IN}};

Lexer::Lexer(std::string_view Src, std::string Filename)
    : Index(0), Line(1), Column(1), Size(Src.length()), Src(Src),
      Filename(Filename) {}

/**
 * Lexes a single token
 */
Token Lexer::lex() {
  // Skip all leading whitespace
  takeWhile(std::isspace);
  if (Index >= Size) {
    return Token{TokenTag::EOI, "", Filename, Line, Column};
  }

  // Keep track of start position
  auto Char = Src[Index];
  auto StartLine = Line;
  auto StartCol = Column;
  switch (Char) {
  case '+':
    step();
    return Token{TokenTag::PLUS, "+", Filename, StartLine, StartCol};
  case '-':
    step();
    return Token{TokenTag::MINUS, "-", Filename, StartLine, StartCol};
  case '*':
    step();
    return Token{TokenTag::STAR, "*", Filename, StartLine, StartCol};
  case '/':
    step();
    return Token{TokenTag::SLASH, "/", Filename, StartLine, StartCol};

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
  throw "Invalid Token";
}

auto Lexer::step() -> char {
  // TODO: Check for out of range
  auto Char = Src[Index];
  Index += 1;
  if (Char == '\n') {
    Line += 1;
    Column = 1;
  } else {
    Column += 1;
  }
  return Char;
}

auto Lexer::takeWhile(std::function<bool(char)> Predicate) -> std::string {
  const auto Len = Src.length();
  auto Start = Index;
  while (Index < Len && Predicate(Src[Index])) {
    step();
  }
  std::string SubStr{Src.substr(Start, Index - Start)};
  return SubStr;
}