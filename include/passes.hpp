#ifndef PASSES_H
#define PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace lettuce {
auto createLowerToLLVMPass() -> std::unique_ptr<mlir::Pass>;
} // namespace lettuce

#endif