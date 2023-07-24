#ifndef TYPE_CHECKER_H
#define TYPE_CHECKER_H

#include "AST.hpp"
#include "lexer.hpp"
#include "types.hpp"

#include <memory>
#include <unordered_map>
#include <utility>

using TypeCtx = std::unordered_map<std::string, Type>;

class TypeError : public UserError {
public:
  TypeError(Location Loc, std::string Message)
      : UserError(Loc.Filename, Loc.Line, Loc.Column,
                  "Type Error: " + Message){};
};

auto typeCheck(TypeCtx Ctx, Expr &Exp, Type Expected) -> void;
auto typeInfer(TypeCtx Ctx, Expr &Exp) -> Type;

#endif