//===- RomaineOps.td - Romaine dialect ops -----------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef ROMAINE_OPS
#define ROMAINE_OPS

include "mlir/IR/OpBase.td"
include "Romaine/RomaineDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Romaine_ConstantOp : Romaine_Op<"const", [Pure]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `romaine.foo` operation illustrates how to define a new
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


#endif // ROMAINE_OPS
