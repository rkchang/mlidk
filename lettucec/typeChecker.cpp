#include "AST.fwd.hpp"
#include "AST.hpp"
#include "typeChecker.hpp"
#include "types.hpp"
#include <iostream>
#include <unordered_set>

auto checkType(Location Loc, std::shared_ptr<Type> Actual, 
							 std::shared_ptr<Type> Expected) -> void {
  // Compare underlying Type value!
  if (*Actual != *Expected) {
    throw TypeError(Loc, 
										"Expected " + Expected->toString() + ", but got " +
										Actual->toString());
  }
}

auto TypeChecker::visit(RootNode &Node, std::any Context) -> std::any {
  Node.Exp->accept(*this, Context);
  return NULL;
}

auto TypeChecker::visit(DefExpr &Node, std::any Context) -> std::any {
  auto *E = static_cast<DefExpr *>(&Node);

  auto DefinitionNames = std::unordered_set<std::string>();
  for (auto &Definition : E->Definitions) {
    // Check for duplicate definition names
    if (DefinitionNames.contains(Definition.Name)) {
      throw TypeError(Node.Loc,
                      "Duplicate definition name '" + Definition.Name + "'");
    }
    DefinitionNames.insert(Definition.Name);

    auto ParamNames = std::unordered_set<std::string>();
    auto ParamTypes = std::vector<Type>();

    // Collect parameter types
    for (auto &Param : Definition.Params) {
      auto ParamName = Param.first;
      auto ParamTy = Param.second;
      if (ParamNames.contains(ParamName)) {
        throw TypeError(Node.Loc,
                        "Duplicate parameter name '" + ParamName + "'");
      }
      ParamNames.insert(ParamName);
      ParamTypes.push_back(ParamTy);
    }

    auto RetTy = std::make_shared<Type>(Definition.ReturnType);
    auto FuncTy = std::make_shared<FuncT>(ParamTypes, RetTy);
    Definition.Ty = FuncTy;
    Ctx[Definition.Name] = FuncTy;
  }

  // Check individual bodies
  for (auto &Definition : E->Definitions) {
    auto ParamNames = std::unordered_set<std::string>();
    // Collect parameter types
    for (auto &Param : Definition.Params) {
      auto ParamName = Param.first;
      auto ParamTy = Param.second;
      ParamNames.insert(ParamName);
      Ctx[ParamName] = std::make_shared<Type>(ParamTy);
    }
    auto RetTy = std::make_shared<Type>(Definition.ReturnType);
		Definition.Body->accept(*this, Context);
    checkType(Definition.Body->Loc, Definition.Body->Ty, RetTy);
    // Remove parameters from context
    for (auto &ParamName : ParamNames) {
      Ctx.erase(ParamName);
    }
  }

	Node.Body->accept(*this, Context);
  E->Ty = Node.Body->Ty;

  return NULL;
}

auto TypeChecker::visit(LetExpr &Node, std::any Context) -> std::any {
  auto *E = static_cast<LetExpr *>(&Node);
  Node.Value->accept(*this, Context);
  Ctx[E->Name] = std::any_cast<std::shared_ptr<Type>>(Node.Value->Ty);
  Node.Body->accept(*this, Context);
  Ctx.erase(E->Name);
  Node.Ty = std::any_cast<std::shared_ptr<Type>>(Node.Body->Ty);
  return NULL;
}

auto TypeChecker::visit(IfExpr &Node, std::any Context) -> std::any {
	Node.Condition->accept(*this, Context);
	checkType(Node.Condition->Loc, Node.Condition->Ty, BoolT);
  Node.TrueBranch->accept(*this, Context);  
  Node.FalseBranch->accept(*this, Context);  
	checkType(Node.TrueBranch->Loc, Node.FalseBranch->Ty, 
						Node.TrueBranch->Ty);
  Node.Ty = Node.TrueBranch->Ty;
  return NULL;
}

auto TypeChecker::visit(BinaryExpr &Node, std::any Context) -> std::any {
  Node.Left->accept(*this, Context);
  Node.Right->accept(*this, Context);
	auto Lhs = Node.Left->Ty;
	auto Rhs = Node.Left->Ty;

	auto Operator = Node.Operator;
  switch (Operator) {
    // Arithmetic
  case TokenOp::OpType::ADD:
  case TokenOp::OpType::MINUS:
  case TokenOp::OpType::MUL:
  case TokenOp::OpType::DIV:
    checkType(Node.Left->Loc, Lhs, Int32T);
    checkType(Node.Right->Loc, Rhs, Int32T);
		Node.Ty = Int32T;
		break;
  // Comparisson
  case TokenOp::OpType::LT:
  case TokenOp::OpType::LE:
  case TokenOp::OpType::GT:
  case TokenOp::OpType::GE:
    checkType(Node.Left->Loc, Lhs, Int32T);
    checkType(Node.Right->Loc, Rhs, Int32T);
		Node.Ty = BoolT;
		break;
  // Boolean
  case TokenOp::OpType::AND:
  case TokenOp::OpType::OR:
    checkType(Node.Left->Loc, Lhs, BoolT);
    checkType(Node.Right->Loc, Rhs, BoolT);
		Node.Ty = BoolT;
		break;
  // Equality
  case TokenOp::OpType::EQ:
  case TokenOp::OpType::NE:
    checkType(Node.Left->Loc,  Rhs, Lhs);
		Node.Ty = BoolT;
		break;
  // Unary
  case TokenOp::OpType::NOT:
    throw TypeError(Node.Left->Loc, "Unsupported operation");
	default:
		std::cerr << "Unknown Optype" << std::endl;
		std::exit(1);
  }
  return NULL;
}

auto TypeChecker::visit(UnaryExpr &Node, std::any Context) -> std::any {
  Node.Right->accept(*this, Context);;
	auto Operator = Node.Operator;

  switch (Operator) {
  case TokenOp::OpType::NOT:
    checkType(Node.Loc, Node.Right->Ty, BoolT);
    return BoolT;
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
    throw TypeError(Node.Right->Loc, "Unsupported operation");
	default:
		std::cerr << "Unknown Optype" << std::endl;
		std::exit(1);
  }

	Node.Ty = Node.Right->Ty;
  return NULL;
}

auto TypeChecker::visit(IntExpr &Node, std::any TypeCtx) -> std::any {
	Node.Ty = Int32T;
  return NULL;
}

auto TypeChecker::visit(BoolExpr &Node, std::any Context) -> std::any {
	Node.Ty = BoolT;
  return NULL;
}

auto TypeChecker::visit(VarExpr &Node, std::any Context) -> std::any {
  if (Ctx.contains(Node.Name)) {
    auto Ty = Ctx[Node.Name];
    Node.Ty = Ty;
    return NULL;
  }
  throw TypeError(Node.Loc, "Undefined variable '" + Node.Name + "'");
  return NULL;
}

auto TypeChecker::visit(CallExpr &Node, std::any Context) -> std::any {
  Node.Func->accept(*this, Context);
	if (Node.Func->Ty->Tag != TypeTag::FUNC) {
		throw TypeError(Node.Loc,
										"Cannot call expression of type " 
										+ Node.Func->Ty->toString());
	}
  auto *FuncTy = static_cast<FuncT *>(Node.Func->Ty.get());
	auto *E = static_cast<CallExpr *>(&Node);
  auto ParamsSize = FuncTy->Params.size();
  auto ArgsSize = E->Args.size();
  if (ParamsSize != ArgsSize) {
    throw TypeError(Node.Loc, "Expected " + std::to_string(ParamsSize) +
                                 " parameters, but got " +
                                 std::to_string(ArgsSize) + " arguments");
  }

	for (size_t Idx = 0; Idx < ParamsSize; Idx++) {
		auto& Arg = Node.Args[Idx];
    Arg->accept(*this, Context);
		checkType(Arg->Loc, Arg->Ty, std::make_shared<Type>(FuncTy->Params[Idx]));
  }

  Node.Ty = FuncTy->Ret;
  return NULL;
}

auto TypeChecker::visit(FuncExpr &Node, std::any Context) -> std::any {
  auto ParamNames = std::unordered_set<std::string>();
  auto ParamTypes = std::vector<Type>();

  // Collect parameter types
  for (auto &Param : Node.Params) {
    auto ParamName = Param.first;
    auto ParamTy = Param.second;
    if (ParamNames.contains(ParamName)) {
      throw TypeError(Node.Loc,
                      "Duplicate parameter name '" + ParamName + "'");
    }
    ParamNames.insert(ParamName);
    ParamTypes.push_back(ParamTy);
    Ctx[ParamName] = std::make_shared<Type>(ParamTy);
  }

  Node.Body->accept(*this, Context);
  auto FuncTy = std::make_shared<FuncT>(ParamTypes, Node.Body->Ty);

  // Remove parameters from context
  for (auto &ParamName : ParamNames) {
    Ctx.erase(ParamName);
  }

  Node.Ty = FuncTy;
  return NULL;
}
