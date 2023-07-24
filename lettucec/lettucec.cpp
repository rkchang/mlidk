#include "ASTPrinter.hpp"
#include "MLIR_gen.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "passes.hpp"

#include <iostream>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input toy file>"),
                                         llvm::cl::init("-"),
                                         llvm::cl::value_desc("filename"));

llvm::cl::opt<bool>
    Dbg("printIR", llvm::cl::desc("Output AST, MLIR and LLVM IR to stdout"));

std::unique_ptr<RootNode> parseInputFile(const llvm::StringRef &Buffer,
                                         const std::string &Filename) {
  // Print source
  if (Dbg) {
    std::cout << "Source:" << std::endl;
    llvm::outs() << Buffer;
  }

  auto Lexr = Lexer(Buffer, Filename);
  auto Parsr = Parser(Lexr);
  auto AST = Parsr.parse();
  // Print out the AST
  if (Dbg) {
    std::cout << std::endl << "AST:" << std::endl;
    auto Printer = ASTPrinter();
    AST->accept(Printer, 0);
  }
  return AST;
}

void runJIT(mlir::MLIRContext &Context, std::unique_ptr<RootNode> AST) {
  auto MLIRGenerator = MLIRGen(Context);
  AST->accept(MLIRGenerator, 0);
  mlir::OwningOpRef<mlir::ModuleOp> Module = MLIRGenerator.Module;
  if (failed(mlir::verify(Module->getOperation()))) {
    std::cout << "Failed to verify MLIR Module\n";
    exit(-1);
  }
  if (Dbg) {
    // Print out the MLIR
    std::cout << std::endl << "MLIR:" << std::endl;
    Module->dump();
  }

  // Lower to LLVM
  auto PM = mlir::PassManager(&Context);
  applyPassManagerCLOptions(PM);
  PM.addPass(lettuce::createLowerToLLVMPass());
  if (mlir::failed(PM.run(*Module))) {
    exit(4);
  }
  if (Dbg) {
    // Print out the LLVM dialect
    std::cout << std::endl << "LLVM Dialect:" << std::endl;
    Module->dump();
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerLLVMDialectTranslation(*Module->getContext());

  // Create an MLIR execution engine. The execution engine eagerly
  // JIT-compiles the module.
  mlir::ExecutionEngineOptions EngineOptions;
  EngineOptions.enableObjectDump = true;
  auto MaybeEngine =
      mlir::ExecutionEngine::create(Module->getOperation(), EngineOptions);
  assert(MaybeEngine && "failed to construct an execution engine");
  auto &Engine = MaybeEngine.get();

  if (InputFilename == "-") {
    Engine->dumpToObjectFile("output.o");
  } else {
    // Invoke the JIT-compiled function.
    int32_t Result = 0;
    llvm::SmallVector<void *> argsArray{&Result};
    auto InvocationResult = Engine->invokePacked("main", argsArray);
    if (InvocationResult) {
      llvm::errs() << "JIT invocation failed\n";
      exit(-1);
    }
    std::cout << "Return value: " << Result << std::endl;
  }
}

int main(int argc, char *argv[]) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "The lettuce compiler\n");

  // Generate MLIR
  mlir::MLIRContext Context;
  Context.getOrLoadDialect<mlir::arith::ArithDialect>();
  Context.getOrLoadDialect<mlir::func::FuncDialect>();
  Context.getOrLoadDialect<mlir::scf::SCFDialect>();

  // TODO: allow input from stdin
  if (InputFilename == "-") {
    // TODO: Use a readline library (maybe libedit?)
    std::string Line;
    std::cout << ">>> ";
    while (std::getline(std::cin, Line)) {
      auto AST = parseInputFile(Line, InputFilename);
      runJIT(Context, std::move(AST));
    }
  } else {
    auto FileOrErr = llvm::MemoryBuffer::getFile(InputFilename);
    if (const std::error_code Ec = FileOrErr.getError()) {
      std::cerr << "Error opening input file: " << Ec.message() << "\n";
      return 1;
    }
    auto Buffer = FileOrErr.get()->getBuffer();
    auto AST = parseInputFile(Buffer, InputFilename);
    runJIT(Context, std::move(AST));
  }
}