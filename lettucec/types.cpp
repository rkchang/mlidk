#include "types.hpp"
#include <string>

auto Type::operator==(Type &That) -> bool {
  switch (this->Tag) {
  case TypeTag::INT32:
  case TypeTag::BOOL:
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
}

auto Type::toString() -> std::string {
  switch (Tag) {
  case TypeTag::INT32:
    return "i32";
  case TypeTag::BOOL:
    return "bool";
  case TypeTag::FUNC: {
    auto *Self = static_cast<FuncT *>(this);
    std::string S = "(";
    for (auto P : Self->Params) {
      S.append(P.toString());
    }
    S.append(") -> " + Self->Ret->toString());
    return S;
  }
  }
}