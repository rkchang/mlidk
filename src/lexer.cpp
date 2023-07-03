#include <lexer.hpp>
#include <map>

// All keywords in language
const std::map<std::string, TokenTag> KeyWords = {{"let", TokenTag::LET},
                                                  {"in", TokenTag::IN}};

Lexer::Lexer(std::string_view Src, std::string Filename)
    : Index(0), Line(1), Column(1), Src(Src), Filename(Filename) {}

/**
 * Lexes a single token
 */
Token Lexer::lex() {
  const auto Len = Src.length();
  // Skip all leading whitespace
  while (Index < Len && std::isspace(Src[Index])) {
    step();
  }
  if (Index >= Len) {
    return Token{TokenTag::EOI, "", Filename, Line, Column};
  }

  // Keep track of start position
  auto Char = Src[Index];
  auto Start = Index;
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
      while (Index < Len && std::isdigit(Char)) {
        step();
      }
      std::string Value{Src.substr(Start, Index - Start)};
      return Token{TokenTag::INT, Value, Filename, StartLine, StartCol};

    } else if (std::isalpha(Char)) {
      // Ident
      while (Index < Len && (std::isalpha(Char) || std::isdigit(Char))) {
        step();
      }
      std::string Value{Src.substr(Start, Index - Start)};
      auto Tag = TokenTag::IDENT;
      if (KeyWords.contains(Value)) {
        Tag = KeyWords.at(Value);
      }
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