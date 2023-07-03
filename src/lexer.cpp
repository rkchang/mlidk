#include <lexer.hpp>

Lexer::Lexer(std::string_view Src, std::string Filename )
    : Src(Src), Filename(Filename) {}

Token Lexer::lex() { return Token{"hello", Filename, 0, 0}; }