#ifndef TYPES_H
#define TYPES_H

#include <cstddef>
#include <string>

enum class TypeTag { INT32, BOOL };

union TypeData {
  std::nullptr_t Null;
};

struct Type {
  TypeTag Tag;
  TypeData Data;

  auto operator==(Type &That) -> bool;
  auto toString() -> std::string;
};

const Type BoolT = Type{TypeTag::BOOL, {nullptr}};
const Type Int32T = Type{TypeTag::INT32, {nullptr}};

#endif