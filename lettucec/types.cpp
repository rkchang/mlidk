#include "types.hpp"
#include <iostream>
#include <string>

auto Type::operator==(Type &That) -> bool {
  switch (Tag) {
  case TypeTag::INT32:
  case TypeTag::BOOL:
  case TypeTag::VOID:
    return this->Tag == That.Tag;
  case TypeTag::FUNC:
    if (this->Tag != That.Tag) {
      return false;
    }
    auto *Self = static_cast<FuncT *>(this);
    auto *Other = static_cast<FuncT *>(&That);
    if (Self->Ret != Other->Ret) {
      return false;
    }
    if (Self->Params.size() != Other->Params.size()) {
      return false;
    }
    for (size_t Idx = 0; Idx < Self->Params.size(); Idx++) {
      if (Self->Params[Idx] != Other->Params[Idx]) {
        return false;
      }
    }
    return true;
  }
  std::cerr << "Invalid TypeTag\n";
  exit(1);
}

auto Type::toString() -> std::string const {
  switch (Tag) {
  case TypeTag::INT32:
    return "i32";
  case TypeTag::BOOL:
    return "bool";
  case TypeTag::VOID:
    return "void";
  case TypeTag::FUNC: {
    auto *Self = static_cast<FuncT *>(this);
    std::string S = "(";
    auto First = true;
    for (auto P : Self->Params) {
      if (!First) {
        S.append(", ");
      }
      S.append(P.toString());
      First = false;
    }
    S.append(") -> " + Self->Ret->toString());
    return S;
  }
  }
  std::cerr << "Invalid TypeTag\n";
  exit(1);
}