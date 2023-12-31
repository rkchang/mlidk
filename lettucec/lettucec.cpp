#include "ASTPrinter.hpp"
#include "MLIR_gen.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "passes.hpp"
#include "type_checker.hpp"
#include "types.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <unordered_map>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

const std::string ReplMode = "REPL";

llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input lettuce file>"),
                                         llvm::cl::init(ReplMode),
                                         llvm::cl::value_desc("filename"));

llvm::cl::opt<bool>
    Dbg("dbg", llvm::cl::desc("Output AST, MLIR and LLVM IR to stdout"));

llvm::cl::opt<bool> DbgSrc("dbg-src",
                           llvm::cl::desc("Output source code to stdout"));

llvm::cl::opt<bool> DbgAst("dbg-ast", llvm::cl::desc("Output AST to stdout"));

llvm::cl::opt<bool> DbgMLIR("dbg-mlir",
                            llvm::cl::desc("Output MLIR to stdout"));

llvm::cl::opt<bool> DbgLLVM("dbg-llvm",
                            llvm::cl::desc("Output LLVM IR to stdout"));

llvm::cl::opt<bool> Canonicalize("canonicalize",
                                 llvm::cl::desc("Canonicalize output"));

std::unique_ptr<RootNode> parseInputFile(const llvm::StringRef &Buffer,
                                         const std::string &Filename) {
  // Print source
  if (Dbg || DbgSrc) {
    std::cout << "Source:" << std::endl;
    llvm::outs() << Buffer << "\n";
  }

  auto Lexr = Lexer(Buffer, Filename);
  auto Parsr = Parser(Lexr);
  auto AST = Parsr.parse();
  // Print out the AST
  if (Dbg || DbgAst) {
    std::cout << std::endl << "AST:" << std::endl;
    auto Printer = ASTPrinter();
    AST->accept(Printer, 0);
  }

  auto TypeCtx = std::unordered_map<std::string, std::shared_ptr<Type>>{};
  FuncT Ftype = {std::vector<Type>{*Int32T}, Int32T};
  TypeCtx.insert({"print", std::make_shared<FuncT>(Ftype)});

  typeInfer(TypeCtx, *(AST->Exp));

  if (Dbg || DbgAst) {
    std::cout << std::endl << "TAST:" << std::endl;
    auto Printer = ASTPrinter();
    AST->accept(Printer, 0);
  }

  return AST;
}

mlir::OwningOpRef<mlir::ModuleOp> genMLIR(mlir::MLIRContext &Context,
                                          std::unique_ptr<RootNode> AST) {
  auto MLIRGenerator = MLIRGen(Context);
  AST->accept(MLIRGenerator, 0);
  mlir::OwningOpRef<mlir::ModuleOp> Module = MLIRGenerator.Module;
  if (failed(mlir::verify(Module->getOperation()))) {
    std::cerr << "Failed to verify MLIR Module\n";
    std::exit(1);
  }

  if (Canonicalize) {
    auto PM0 = mlir::PassManager(&Context);
    applyPassManagerCLOptions(PM0);
    PM0.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(PM0.run(*Module))) {
      std::cerr << "Failed to canonicalize dialect\n";
      std::exit(1);
    }
  }

  if (Dbg || DbgMLIR) {
    // Print out the MLIR
    std::cout << std::endl << "MLIR:" << std::endl;
    Module->dump();
  }

  // Lower to LLVM
  auto PM = mlir::PassManager(&Context);
  applyPassManagerCLOptions(PM);
  PM.addPass(lettuce::createLowerToLLVMPass());

  // Other passes may independently create unrealized_cast operations.
  // Reconcile them once at the end
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(PM.run(*Module))) {
    std::cerr << "Failed to lower to LLVM dialect\n";
    std::exit(1);
  }
  if (Dbg || DbgLLVM) {
    // Print out the LLVM dialect
    std::cout << std::endl << "LLVM Dialect:" << std::endl;
    Module->dump();
  }
  return Module;
}

void runJIT(mlir::OwningOpRef<mlir::ModuleOp> Module) {
  // Initialize LLVM targets.
  mlir::registerLLVMDialectTranslation(*Module->getContext());

  if (InputFilename == ReplMode) {
    // Run JIT
    // Create an MLIR execution engine.
    mlir::ExecutionEngineOptions EngineOptions;
    EngineOptions.enableObjectDump = true;
    auto MaybeEngine =
        mlir::ExecutionEngine::create(Module->getOperation(), EngineOptions);
    assert(MaybeEngine && "Failed to construct an execution engine");
    auto &Engine = MaybeEngine.get();

    // Invoke the JIT-compiled function.
    int32_t Result = 0;
    llvm::SmallVector<void *> ArgsArray{&Result};
    auto InvocationResult = Engine->invokePacked("main", ArgsArray);
    if (InvocationResult) {
      std::cerr << "JIT invocation failed\n";
      std::exit(1);
    }
    std::cout << "Return value: " << Result << std::endl;
  } else {
    // Lower MLIR to LLVM IR and emit object code
    // Taken from the kaleidoscope tutorial with some minor edits:
    // https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/LangImpl08.html

    // Convert dialect to LLVM IR
    llvm::LLVMContext LlvmContext;
    auto LlvmModule =
        mlir::translateModuleToLLVMIR(Module->getOperation(), LlvmContext);
    if (!LlvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      std::exit(1);
    }
    // Print out the LLVM IR
    if (Dbg) {
      std::cout << std::endl << "LLVM IR:" << std::endl;
      LlvmModule->dump();
    }

    // Setup the target
    auto TargetTriple = llvm::sys::getDefaultTargetTriple();
    LlvmModule->setTargetTriple(TargetTriple);
    std::string Error;
    const auto *Target =
        llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.
    if (!Target) {
      llvm::errs() << Error;
      std::exit(1);
    }
    const auto *CPU = "generic";
    const auto *Features = "";
    llvm::TargetOptions Opt;
    auto RM = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
    auto TheTargetMachine = std::unique_ptr<llvm::TargetMachine>(
        Target->createTargetMachine(TargetTriple, CPU, Features, Opt, RM));
    LlvmModule->setDataLayout(TheTargetMachine->createDataLayout());

    // Create object file
    const auto *Filename = "output.o";
    std::error_code EC;
    llvm::raw_fd_ostream Dest(Filename, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Could not open file: " << EC.message();
      std::exit(1);
    }

    // Emit object code
    llvm::legacy::PassManager Pass;
    auto FileType = llvm::CGFT_ObjectFile;
    if (TheTargetMachine->addPassesToEmitFile(Pass, Dest, nullptr, FileType)) {
      llvm::errs() << "TheTargetMachine can't emit a file of this type";
      std::exit(1);
    }
    Pass.run(*LlvmModule);
    Dest.flush();
    std::cout << "Wrote output.o" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::cl::ParseCommandLineOptions(argc, argv, "The lettuce compiler\n");

  // Generate MLIR
  mlir::MLIRContext Context;
  Context.getOrLoadDialect<mlir::arith::ArithDialect>();
  Context.getOrLoadDialect<mlir::func::FuncDialect>();
  Context.getOrLoadDialect<mlir::scf::SCFDialect>();
  Context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  if (InputFilename == ReplMode) {
    std::string Line;
    std::cout << ">>> ";
    while (std::getline(std::cin, Line)) {
      try {
        auto AST = parseInputFile(Line, InputFilename);
        runJIT(genMLIR(Context, std::move(AST)));
      } catch (UserError &Excep) {
        std::cerr << Excep.what() << "\n";
      }
      std::cout << ">>> ";
    }
  } else {
    auto FileOrErr = llvm::MemoryBuffer::getFile(InputFilename);
    if (const std::error_code Ec = FileOrErr.getError()) {
      std::cerr << "Error opening input file: " << Ec.message() << "\n";
      return 1;
    }
    auto Buffer = FileOrErr.get()->getBuffer();
    auto AST = parseInputFile(Buffer, InputFilename);
    auto GeneratedMLIR = genMLIR(Context, std::move(AST));
    runJIT(std::move(GeneratedMLIR));
  }
}