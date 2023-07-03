#include <string>

struct Token {
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
  std::string_view Src;
  std::string Filename;
};