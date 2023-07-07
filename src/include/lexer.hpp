#pragma once

#include <exception>
#include <functional>
#include <string>

enum class TokenTag {
  INT,
  IDENT,
  PLUS,
  MINUS,
  STAR,
  SLASH,
  LET,
  IN,
  EQUAL,
  EOI, // Might not be present
};

class LexerError : public std::exception {
public:
  const int Line;
  const int Column;
  const std::string Message;
  const std::string Filename;

  LexerError(int Line, int Column, std::string Message, std::string Filename);
  auto message() const -> const std::string;
  auto what() const noexcept -> const char *;
};

class InvalidToken : public LexerError {
public:
  InvalidToken(int Line, int Column, std::string Filename);
};

struct Token {
  const TokenTag Tag;
  const std::string Value;
  const std::string Filename;
  const int Line;
  const int Column;
};

class Lexer {
public:
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
   * Returns true if input has been fully consumed
   */
  auto isDone() -> bool;

private:
  size_t Index;
  int Line;
  int Column;
  const size_t Size;
  const std::string_view Source;
  const std::string Filename;

  /**
   * Steps over a single character
   */
  auto step() -> char;

  /**
   * Consumes input while predicate is true
   */
  auto takeWhile(std::function<bool(char)> Predicate) -> std::string;
};