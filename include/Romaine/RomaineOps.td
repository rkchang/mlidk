//===- RomaineOps.td - Romaine dialect ops -----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef ROMAINE_OPS
#define ROMAINE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "Romaine/RomaineDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Romaine_ConstantOp : Romaine_Op<"const", [Pure]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `romaine.foo` operation illustrates how to define a new
    }];

    let arguments = (ins I32Attr:$value);
    let results = (outs I32:$res);
      let builders = [
        // Build a constant with a given constant floating-point value.
        OpBuilder<(ins "int":$value)>
      ];
}

def Romaine_AddOp : Romaine_Op<"add", [Pure]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `romaine.foo` operation illustrates how to define a new
    }];

    let arguments = (ins I32:$left, I32:$right);
    let results = (outs I32:$res);
}

def Romaine_MulOp : Romaine_Op<"mul", [Pure]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `romaine.foo` operation illustrates how to define a new
    }];

    let arguments = (ins I32:$left, I32:$right);
    let results = (outs I32:$res);
}


#endif // ROMAINE_OPS
