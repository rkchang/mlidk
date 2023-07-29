#include "MLIR_gen.hpp"
#include "AST.hpp"
#include "lexer.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <any>
#include <string>
#include <vector>

// Return a value representing an access into a global string with the given
// name, creating the string if necessary.
// Taken from the toy tutorial with some minor edits
// https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/
auto MLIRGen::getOrCreateGlobalString(mlir::Location Loc, llvm::StringRef Name,
                                      mlir::StringRef Value) -> mlir::Value {
  // Create the global at the entry of the module.
  mlir::LLVM::GlobalOp Global;
  if (!(Global = Module.lookupSymbol<mlir::LLVM::GlobalOp>(Name))) {
    // Reset insertion point when this goes out of scope
    const mlir::OpBuilder::InsertionGuard InsertGuard(Buildr);
    Buildr.setInsertionPointToStart(Module.getBody());
    auto Type =
        mlir::LLVM::LLVMArrayType::get(Buildr.getI8Type(), Value.size());
    // Create a global that contains Value
    Global = Buildr.create<mlir::LLVM::GlobalOp>(
        Loc, Type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, Name,
        Buildr.getStringAttr(Value),
        /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  const mlir::Value GlobalPtr =
      Buildr.create<mlir::LLVM::AddressOfOp>(Loc, Global);
  const mlir::Value Cst0 = Buildr.create<mlir::LLVM::ConstantOp>(
      Loc, Buildr.getI64Type(), Buildr.getIndexAttr(0));
  return Buildr.create<mlir::LLVM::GEPOp>(
      Loc,
      mlir::LLVM::LLVMPointerType::get(
          mlir::IntegerType::get(Buildr.getContext(), 8)),
      GlobalPtr, llvm::ArrayRef<mlir::Value>({Cst0, Cst0}));
}

MLIRGen::Error::Error(Location Loc, std::string Msg)
    : std::runtime_error(Loc.Filename + ":" + std::to_string(Loc.Line) + ":" +
                         std::to_string(Loc.Column) + ": " +
                         "MLIR Generator Error" + ": " + Msg),
      Loc(Loc) {}

MLIRGen::MLIRGen(mlir::MLIRContext &Context) : Buildr(&Context) {
  Module = mlir::ModuleOp::create(Buildr.getUnknownLoc());
  Buildr.setInsertionPointToEnd(Module.getBody());
}

auto MLIRGen::loc(const Location &Loc) -> mlir::Location {
  return mlir::FileLineColLoc::get(Buildr.getStringAttr(Loc.Filename), Loc.Line,
                                   Loc.Column);
}

auto MLIRGen::freshName(std::string Prefix) -> std::string {
  auto Id = NextId++;
  return Prefix + std::to_string(Id);
}

// Type converters

auto mlirType(Type &Ty, mlir::OpBuilder Buildr) -> mlir::Type;

auto mlirFunctionType(FuncT &Ty, mlir::OpBuilder Buildr) -> mlir::FunctionType {
  auto Params = std::vector<mlir::Type>();
  for (auto &Param : Ty.Params) {
    Params.push_back(mlirType(Param, Buildr));
  }
  auto Ret = mlirType(*Ty.Ret, Buildr);
  return Buildr.getFunctionType(Params, Ret);
}

auto mlirType(Type &Ty, mlir::OpBuilder Buildr) -> mlir::Type {
  switch (Ty.Tag) {
  case TypeTag::INT32:
    return Buildr.getI32Type();
  case TypeTag::BOOL:
    return Buildr.getI1Type();
  case TypeTag::FUNC: {
    auto *T = static_cast<FuncT *>(&Ty);
    return mlirFunctionType(*T, Buildr);
  }
  case TypeTag::VOID:
    throw "Unsupported";
  }
}

//

auto MLIRGen::visit(const RootNode &Node, std::any Context) -> std::any {
  auto Loc = loc(Node.Loc);
  const llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> Scope(
      SymbolTable);

  // Create a function declaration for printf, the signature is:
  //   i32 printf(i8*, ...)`
  // First arg is the format str, varargs are the variables
  auto LlvmFnType = mlir::LLVM::LLVMFunctionType::get(
      Buildr.getI32Type(), mlir::LLVM::LLVMPointerType::get(Buildr.getI8Type()),
      /*isVarArg=*/true);
  Buildr.create<mlir::LLVM::LLVMFuncOp>(loc(Node.Loc), "printf", LlvmFnType);

  auto RetTy = mlirType(*(Node.Exp->Ty), Buildr);
  auto Ty = Buildr.getFunctionType(std::nullopt, {RetTy});

  auto Fun = Buildr.create<mlir::func::FuncOp>(Loc, "main", Ty);
  Fun.addEntryBlock();
  auto &Blk = Fun.front();
  Buildr.setInsertionPointToStart(&Blk);

  auto V = Node.Exp->accept(*this, Context);

  auto Value = std::any_cast<mlir::Value>(V);
  Buildr.create<mlir::func::ReturnOp>(Loc, Value);

  return Fun;
}

auto MLIRGen::visit(const DefExpr &Node, std::any Context) -> std::any {
  const auto Scope =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>(SymbolTable);

  auto AllDefinitions =
      std::vector<std::pair<std::string, mlir::FunctionType>>();

  // A DefExpr represents a mutually recusive group of functions, and each
  // function must be visible from every other function as well as the body
  // expression at the end. To do this we first iterate over all the function
  // definitions and:
  //  - Collect the function names and their types, to be used later when
  //    generating individual function bodies
  //  - Generate the function "constant" operation for the final body
  //    expression, and insert it to the SymbolTable
  // This approach means that every function call every other function
  // (including itself) indirectly. But we're going with it because it is very
  // simple to implement
  for (auto &Definition : Node.Definitions) {
    auto *T = static_cast<FuncT *>(Definition.Ty.get());
    auto FuncTy = mlirFunctionType(*T, Buildr);

    AllDefinitions.push_back(std::make_pair(Definition.Name, FuncTy));

    auto Func = static_cast<mlir::Value>(Buildr.create<mlir::func::ConstantOp>(
        loc(Node.Body->Loc), FuncTy, Definition.Name));

    SymbolTable.insert(Definition.Name, Func);
  }

  // After all the function signatures have been collected, we just need to
  // generate the body for each function. And pass down the signatures of the
  // other functions. (ideally we'd remove the function being generated, but we
  // don't for simplicity)
  for (auto &Definition : Node.Definitions) {
    auto *T = static_cast<FuncT *>(Definition.Ty.get());
    auto FuncTy = mlirFunctionType(*T, Buildr);

    generateFunction(Definition.Loc, Context, Definition.Name, FuncTy,
                     Definition.Params, *Definition.Body, AllDefinitions);
  }

  // Generate body expression
  return Node.Body->accept(*this, Context);
}

auto MLIRGen::visit(const LetExpr &Node, std::any Context) -> std::any {
  const llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> Scope(
      SymbolTable);
  auto V = Node.Value->accept(*this, Context);
  auto Value = std::any_cast<mlir::Value>(V);
  SymbolTable.insert(Node.Name, Value);
  return Node.Body->accept(*this, Context);
}

auto MLIRGen::visit(const IfExpr &Node, std::any Context) -> std::any {
  auto TR = mlir::TypeRange(mlirType(*Node.Ty, Buildr));

  // Compile condition first
  auto Cond =
      std::any_cast<mlir::Value>(Node.Condition->accept(*this, Context));

  // The 'if' builders are pretty knarly and poorly documented, this was mostly
  // discovered through trial and error. This particular constructor should be:
  // TypeRange for result types; Value for condition, bool for withElseRegion
  auto If = Buildr.create<mlir::scf::IfOp>(loc(Node.Loc), TR, Cond, true);

  // Store the current OpBuilder
  auto OldBuildr = Buildr;

  // Both branches follow the same general shape:
  // - Get the respective region from the IfOp
  // - Create a new OpBuilder for the region
  // - Generate the result value for the branch
  // - Generate a YieldOp with the result for the branch

  // Then branch
  auto *Then = &If.getThenRegion();
  Buildr = mlir::OpBuilder(Then);
  auto TrueValue =
      std::any_cast<mlir::Value>(Node.TrueBranch->accept(*this, Context));
  Buildr.create<mlir::scf::YieldOp>(loc(Node.TrueBranch->Loc), TrueValue);

  // Else branch
  auto *Else = &If.getElseRegion();
  Buildr = mlir::OpBuilder(Else);
  auto FalseValue =
      std::any_cast<mlir::Value>(Node.FalseBranch->accept(*this, Context));
  Buildr.create<mlir::scf::YieldOp>(loc(Node.FalseBranch->Loc), FalseValue);

  // Restore old OpBuilder
  Buildr = OldBuildr;

  // For some reson IfOp is not a Value, so we must get the result value
  // for the op, and use that as our return value.
  auto Result = If->getOpResult(0);

  return static_cast<mlir::Value>(Result);
}

auto MLIRGen::visit(const BinaryExpr &Node, std::any Context) -> std::any {
  auto Lhs = std::any_cast<mlir::Value>(Node.Left->accept(*this, Context));
  auto Rhs = std::any_cast<mlir::Value>(Node.Right->accept(*this, Context));
  auto DataType = mlirType(*Node.Ty, Buildr);
  switch (Node.Operator) {
  case TokenOp::OpType::ADD:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::AddIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::MUL:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::MulIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::MINUS:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::SubIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::DIV:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::DivSIOp>(loc(Node.Loc), DataType, Lhs, Rhs));
  case TokenOp::OpType::EQ:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::eq, Lhs, Rhs));
  case TokenOp::OpType::NE:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::ne, Lhs, Rhs));
  case TokenOp::OpType::LT:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::slt, Lhs, Rhs));
  case TokenOp::OpType::LE:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::sle, Lhs, Rhs));
  case TokenOp::OpType::GT:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::sgt, Lhs, Rhs));
  case TokenOp::OpType::GE:
    return static_cast<mlir::Value>(Buildr.create<mlir::arith::CmpIOp>(
        loc(Node.Loc), mlir::arith::CmpIPredicate::sge, Lhs, Rhs));
  case TokenOp::OpType::AND:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::AndIOp>(loc(Node.Loc), Lhs, Rhs));
  case TokenOp::OpType::OR:
    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::OrIOp>(loc(Node.Loc), Lhs, Rhs));

  case TokenOp::OpType::NOT:
    throw Error(Node.Loc, "Unsupported operation");
  }
  throw Error(Node.Loc, "Unknown binary operator");
}

auto MLIRGen::visit(const UnaryExpr &Node, std::any Context) -> std::any {
  switch (Node.Operator) {
  case TokenOp::OpType::NOT: {
    auto Rhs = std::any_cast<mlir::Value>(Node.Right->accept(*this, Context));
    // TODO: Fix this mess
    auto DataType = Buildr.getI1Type();
    auto TrueAttr = Buildr.getBoolAttr(true);
    auto FalseAttr = Buildr.getBoolAttr(false);
    auto True = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                       TrueAttr);
    auto False = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                        FalseAttr);

    return static_cast<mlir::Value>(
        Buildr.create<mlir::arith::SelectOp>(loc(Node.Loc), Rhs, False, True));
  }

  case TokenOp::OpType::ADD:
  case TokenOp::OpType::MINUS:
  case TokenOp::OpType::MUL:
  case TokenOp::OpType::DIV:
  case TokenOp::OpType::EQ:
  case TokenOp::OpType::NE:
  case TokenOp::OpType::LT:
  case TokenOp::OpType::LE:
  case TokenOp::OpType::GT:
  case TokenOp::OpType::GE:
  case TokenOp::OpType::AND:
  case TokenOp::OpType::OR:
    throw Error(Node.Loc, "Unsupported operation");
  }
}

auto MLIRGen::visit(const IntExpr &Node, std::any) -> std::any {
  auto DataType = Buildr.getI32Type();
  auto DataAttribute = Buildr.getI32IntegerAttr(Node.Value);
  auto Op = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                   DataAttribute);
  return static_cast<mlir::Value>(Op);
}

auto MLIRGen::visit(const BoolExpr &Node, std::any) -> std::any {
  auto DataType = Buildr.getI1Type();
  auto DataAttribute = Buildr.getBoolAttr(Node.Value);
  auto Op = Buildr.create<mlir::arith::ConstantOp>(loc(Node.Loc), DataType,
                                                   DataAttribute);
  return static_cast<mlir::Value>(Op);
}

auto MLIRGen::visit(const VarExpr &Node, std::any) -> std::any {
  if (auto Variable = SymbolTable.lookup(Node.Name)) {
    return Variable;
  }
  throw Error(Node.Loc, "Unknown variable reference: " + Node.Name);
}

// Generate code for every argument and add their value to ArgsOut
auto MLIRGen::AddArgs(const std::vector<std::unique_ptr<Expr>> &Args,
                      std::any Context, std::vector<mlir::Value> &ArgsOut)
    -> void {
  for (auto &Arg : Args) {
    auto Res = std::any_cast<mlir::Value>(Arg->accept(*this, Context));
    ArgsOut.push_back(Res);
  }
}

auto MLIRGen::visit(const CallExpr &Node, std::any Context) -> std::any {
  std::vector<mlir::Value> Args;
  // Check if it's a call to print
  if (auto *Var = dynamic_cast<VarExpr *>(Node.Func.get())) {
    if (Var->Name == "print") {
      // Get the format string "%i \n" for printf
      const mlir::Value FormatSpecifierCst = getOrCreateGlobalString(
          loc(Node.Loc), "frmt_spec", mlir::StringRef("%i \n\0", 4));
      Args.push_back(FormatSpecifierCst);
      AddArgs(Node.Args, Context, Args);
      auto CallOp = Buildr.create<mlir::LLVM::CallOp>(
          loc(Node.Loc), Buildr.getI32Type(),
          mlir::SymbolRefAttr::get(Buildr.getContext(), "printf"),
          mlir::ValueRange(Args));
      return CallOp.getResult();
    }
  }
  auto Func = std::any_cast<mlir::Value>(Node.Func->accept(*this, Context));
  AddArgs(Node.Args, Context, Args);
  auto VR = mlir::ValueRange(Args);

  // TODO: Call direct?
  // We call the function indirectly because we are storing it in an SSA
  // variable using a func.constant operation
  auto Call =
      Buildr.create<mlir::func::CallIndirectOp>(loc(Node.Loc), Func, VR);

  return static_cast<mlir::Value>(Call->getResult(0));
}

auto MLIRGen::visit(const FuncExpr &Node, std::any Context) -> std::any {
  // Note how we need to generate a mlir::FunctionType, not mlir::Type,
  // otherwise the overload resolution for FuncOp.build won't work
  auto *T = static_cast<FuncT *>(Node.Ty.get());
  auto FuncTy = mlirFunctionType(*T, Buildr);

  auto Name = freshName("func$");

  generateFunction(Node.Loc, Context, Name, FuncTy, Node.Params, *Node.Body);

  // Create a reference to the function we just created
  auto Res =
      Buildr.create<mlir::func::ConstantOp>(loc(Node.Body->Loc), FuncTy, Name);

  return static_cast<mlir::Value>(Res);
}

auto MLIRGen::generateFunction(
    Location Loca, std::any Context, std::string Name,
    mlir::FunctionType FuncTy, std::vector<std::pair<std::string, Type>> Params,
    Expr &Body,
    std::vector<std::pair<std::string, mlir::FunctionType>> OtherDefinitions)
    -> void {
  // For some reason we have to generate functions at the module top-level
  auto IP = Buildr.saveInsertionPoint();
  Buildr.setInsertionPointToStart(Module.getBody());

  // Store all function parameters as typed and named attributes
  auto ParamsTy = std::vector<mlir::NamedAttribute>();
  for (auto &Param : Params) {
    auto Name = Param.first;
    auto Ty = Param.second;
    auto ParamName = Buildr.getStringAttr(Name);
    auto ParamTy = mlirType(Ty, Buildr);
    auto ParamTyAttr = mlir::TypeAttr::get(ParamTy);
    ParamsTy.push_back(mlir::NamedAttribute(ParamName, ParamTyAttr));
  }
  auto ParamsTyAttr = mlir::ArrayRef<mlir::NamedAttribute>(ParamsTy);

  // Create a FuncOp with a fresh name
  auto Loc = loc(Loca);
  auto Func =
      Buildr.create<mlir::func::FuncOp>(Loc, Name, FuncTy, ParamsTyAttr);

  // Create new scope for function body
  auto Scope =
      llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value>(SymbolTable);

  // Create function body, and move insertion point
  Func.addEntryBlock();
  auto *FuncBody = &Func.getBody();
  Buildr.setInsertionPointToStart(&FuncBody->front());

  // Insert loads for the other functions in the mutually recursive group,
  // if applicable
  for (auto &OtherDefinition : OtherDefinitions) {
    auto Func = static_cast<mlir::Value>(Buildr.create<mlir::func::ConstantOp>(
        Loc, OtherDefinition.second, OtherDefinition.first));

    SymbolTable.insert(OtherDefinition.first, Func);
  }

  // Insert every parameter into the symbol table
  auto Idx = 0;
  for (auto &Param : Params) {
    // For some reason ParamName must be initialized as a llvm::StringRef
    // (instead of using a implicit conversion like in LetExpr)
    // otherwise we get an AddressSanitizer error
    auto ParamName = llvm::StringRef(Param.first);
    auto ParamVal = static_cast<mlir::Value>(Func.getArgument(Idx));
    SymbolTable.insert(ParamName, ParamVal);
    Idx++;
  }

  // Generate body, and insert a ReturnOp with the resulting value
  auto RetVal = std::any_cast<mlir::Value>(Body.accept(*this, Context));
  Buildr.create<mlir::func::ReturnOp>(loc(Body.Loc), RetVal);

  // Restore insertion point
  Buildr.restoreInsertionPoint(IP);
}