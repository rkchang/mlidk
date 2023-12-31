#ifndef TYPES_H
#define TYPES_H

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

enum class TypeTag { VOID, INT32, BOOL, FUNC };

struct Type {
  const TypeTag Tag;
  Type(TypeTag Tag) : Tag(Tag) {}
  virtual ~Type() = default;

  auto operator==(Type &That) -> bool;
  auto toString() -> std::string const;
};

struct VoidTClass : public Type {
  VoidTClass() : Type(TypeTag::VOID) {}
};

struct Int32TClass : public Type {
  Int32TClass() : Type(TypeTag::INT32) {}
};

struct BoolTClass : public Type {
  BoolTClass() : Type(TypeTag::BOOL) {}
};

const std::shared_ptr<Type> VoidT = std::make_shared<VoidTClass>();
const std::shared_ptr<Type> Int32T = std::make_shared<Int32TClass>();
const std::shared_ptr<Type> BoolT = std::make_shared<BoolTClass>();

struct FuncT : public Type {
  std::vector<Type> Params;
  std::shared_ptr<Type> Ret;

  FuncT(std::vector<Type> Params, std::shared_ptr<Type> Ret)
      : Type(TypeTag::FUNC), Params(Params), Ret(Ret) {}
};

#endif