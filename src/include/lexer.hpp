#include <functional>
#include <string>

enum TokenTag {
  INT,
  IDENT,
  PLUS,
  MINUS,
  STAR,
  SLASH,
  LET,
  IN,
  EQUAL,
  EOI,
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
  Lexer(std::string_view Src, std::string Filename);

  /**
   * Lexes a single Token
   */
  Token lex();

private:
  size_t Index;
  int Line;
  int Column;
  const size_t Size;
  const std::string_view Src;
  const std::string Filename;

  /**
   * Steps over a single character
   */
  auto step() -> char;

  /**
   * Consumes input while predicate is true
   */
  auto takeWhile(std::function<bool(char)>) -> std::string;
};