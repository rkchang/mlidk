#include <format>
#include <lexer.hpp>
#include <string>
#include <unordered_map>

// All keywords in language
const std::unordered_map<std::string, TokenTag> KeyWords = {
    {"let", TokenTag::LET}, {"in", TokenTag::IN}};

//------------
// LexerError
//------------

LexerError::LexerError(int Line, int Column, std::string Message,
                       std::string Filename)
    : Line(Line), Column(Column), Message(Message), Filename(Filename) {}

auto LexerError::message() const -> const std::string {
  std::string S = Filename + ":" + std::to_string(Line) + ":" +
                  std::to_string(Column) + ": " + Message;
  return S;
}

auto LexerError::what() const noexcept -> const char * {
  static std::string Msg = message();
  return const_cast<char *>(Msg.c_str());
}

InvalidToken::InvalidToken(int Line, int Column, std::string Filename)
    : LexerError(Line, Column, "Invalid Token", Filename) {}

//--------
// Lexer
//--------

Lexer::Lexer(std::string_view Source, std::string Filename)
    : Index(0), Line(1), Column(1), Size(Source.length()), Source(Source),
      Filename(Filename) {}

auto Lexer::token() -> Token {
  // Skip all leading whitespace
  takeWhile(std::isspace);
  if (isDone()) {
    return Token{TokenTag::EOI, "", Filename, Line, Column};
  }

  // Keep track of start position
  auto Char = Source[Index];
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
  case '=':
    step();
    return Token{TokenTag::EQUAL, "=", Filename, StartLine, StartCol};
  case '(':
    step();
    return Token{TokenTag::LPAREN, "=", Filename, StartLine, StartCol};
  case ')':
    step();
    return Token{TokenTag::RPAREN, "=", Filename, StartLine, StartCol};

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
  throw InvalidToken(Line, Column, Filename);
}

auto Lexer::peek() -> Token {
  // This function should be const,
  // but can't be because token() isn't.
  auto OldIndex = Index;
  auto OldLine = Line;
  auto OldColumn = Column;
  auto Token = token();
  Index = OldIndex;
  Line = OldLine;
  Column = OldColumn;
  return Token;
}

auto Lexer::isDone() -> bool { return Index >= Size; }

auto Lexer::step() -> char {
  // TODO: Check for out of range
  auto Char = Source[Index];
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
  auto Start = Index;
  while (!isDone() && Predicate(Source[Index])) {
    step();
  }
  std::string SubStr{Source.substr(Start, Index - Start)};
  return SubStr;
}