#pragma once

#include <exception>
#include <functional>
#include <optional>
#include <string>

enum class TokenTag {
  // Operators
  PLUS = 0,
  MINUS,
  STAR,
  SLASH,
  EQUAL_EQUAL,
  BANG_EQUAL,
  LESS,
  LESS_EQUAL,
  GREATER,
  GREATER_EQUAL,
  AND,
  OR,
  NOT,
  // Not operators
  EQUAL,
  LPAREN,
  RPAREN,
  COMMA,
  ARROW,
  PIPE,
  DOT,
  COLON,
  // Keywords
  LET,
  IN,
  IF,
  THEN,
  ELSE,
  FUN,
  // Types
  I32,
  BOOL_KW,
  VOID,
  // Identifiers and literals
  INT,
  IDENT,
  BOOL,
  EOI, // Might not be present
};

namespace TokenOp {
enum class OpType : char {
  ADD = 0,
  MINUS,
  MUL,
  DIV,
  EQ,
  NE,
  LT,
  LE,
  GT,
  GE,
  AND,
  OR,
  NOT
};

auto TagToOp(TokenTag) -> OpType;
auto OpToStr(OpType) -> std::string;

} // namespace TokenOp

struct Token {
  const TokenTag Tag;
  const std::string Value;
  const std::string Filename;
  const int Line;
  const int Column;
};

/**
 * Thrown when the user has made an error (ex: an unknown token)
 */
class UserError : public std::runtime_error {
public:
  UserError(std::string Filename, int Line, int Column, std::string Message)
      : std::runtime_error(Filename + ":" + std::to_string(Line) + ":" +
                           std::to_string(Column) + ": " + Message) {}
};

class Lexer {
public:
  class Error : public UserError {
  public:
    Error(int Line, int Column, std::string Message, std::string Filename)
        : UserError(Filename, Line, Column, "Lexer Error: " + Message){};
  };

  struct SrcLoc {
    size_t Index;
    int Line;
    int Column;
  };

  Lexer(std::string_view Source, std::string Filename);

  /**
   * Lexes a single Token, returns EOI if Lexer is done
   */
  auto token() -> Token;

  /**
   * Peeks a single Token ahead
   */
  auto peek() -> Token;

  /**
   * Returns the current state of the Lexer
   */
  auto getState() -> SrcLoc const;

  /**
   * Sets the current state of the Lexer
   */
  auto setState(SrcLoc NewState) -> void;

  /**
   * Returns true if input has been fully consumed
   */
  auto isDone() -> bool;

private:
  SrcLoc State;
  const size_t Size;
  const std::string_view Source;
  const std::string Filename;

  auto lookahead(int N) -> std::optional<char>;

  /**
   * Steps over a single character
   */
  auto step() -> char;

  /**
   * Consumes input while predicate is true
   */
  auto takeWhile(std::function<bool(char)> Predicate) -> std::string;
};