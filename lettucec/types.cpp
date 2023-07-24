#include "types.hpp"
#include <string>

auto Type::operator==(Type &That) -> bool {
  switch (this->Tag) {
  case TypeTag::INT32:
  case TypeTag::BOOL:
    return this->Tag == That.Tag;
  }
}

auto Type::toString() -> std::string {
  switch (Tag) {
  case TypeTag::INT32:
    return "Int32";
  case TypeTag::BOOL:
    return "Bool";
  }
}