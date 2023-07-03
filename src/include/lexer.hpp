#include <cstddef>
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
  Token lex();

private:
  size_t Index;
  int Line;
  int Column;
  const std::string_view Src;
  const std::string Filename;

  auto step() -> char;
};