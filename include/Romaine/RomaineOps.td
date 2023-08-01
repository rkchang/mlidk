//===- RomaineOps.td - Romaine dialect ops -----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef ROMAINE_OPS
#define ROMAINE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "Romaine/RomaineDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Romaine_ConstantOp : Romaine_Op<"const", [Pure]> {
    let summary = "A constant integer value";
    let description = [{
        Represents a constant integer value
    }];

    let arguments = (ins I32Attr:$value);
    let results = (outs I32:$res);
}

def Romaine_AddOp : Romaine_Op<"add", [Pure]> {
    let summary = "Add two numbers";
    let description = [{
        Takes two integers and returns their sum.
    }];

    let arguments = (ins I32:$left, I32:$right);
    let results = (outs I32:$res);
}

def Romaine_MulOp : Romaine_Op<"mul", [Pure]> {
    let summary = "Multiply two numbers";
    let description = [{
        Takes two integers and returns their product.
    }];

    let arguments = (ins I32:$left, I32:$right);
    let results = (outs I32:$res);
}

def Romaine_SubOp : Romaine_Op<"sub", [Pure]> {
    let summary = "Subtract two numbers";
    let description = [{
        Takes two integers and returns their difference.
    }];

    let arguments = (ins I32:$left, I32:$right);
    let results = (outs I32:$res);
}

def Romaine_DivOp : Romaine_Op<"div", [Pure]> {
    let summary = "Divide two numbers";
    let description = [{
        Takes two integers and returns their quotient.
    }];

    let arguments = (ins I32:$left, I32:$right);
    let results = (outs I32:$res);
}

def Romaine_ClosureOp : Romaine_Op<"closure", [FunctionOpInterface, CallableOpInterface]> {
    let summary = "A closure.";
    let description = [{
        A closure.
    }];

    let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                         OptionalAttr<DictArrayAttr>:$arg_attrs,
                         OptionalAttr<DictArrayAttr>:$res_attrs);
    let results = (outs FunctionType:$res);
    let regions = (region AnyRegion:$body);

    // let builders = [OpBuilder<(ins
    //     "FunctionType":$type,
    //     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    //     CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    // >];

    let extraClassDeclaration = [{
        // static ClosureOp create(Location location, FunctionType type,
        //                         ArrayRef<NamedAttribute> attrs = {});

        //===------------------------------------------------------------------===//
        // CallableOpInterface
        //===------------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

        /// Returns the argument attributes for all callable region arguments or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableArgAttrs() {
        return getArgAttrs().value_or(nullptr);
        }

        /// Returns the result attributes for all callable region results or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableResAttrs() {
        return getResAttrs().value_or(nullptr);
        }

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];
}


#endif // ROMAINE_OPS
