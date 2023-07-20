//===- RomaineDialect.td - Romaine dialect -----------*- tablegen -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef ROMAINE_DIALECT
#define ROMAINE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Romaine dialect definition.
//===----------------------------------------------------------------------===//

def Romaine_Dialect : Dialect {
    let name = "romaine";
    let summary = "The highest level dialect in the lettuce hierarchy.";
    let description = [{
        TODO
    }];
    let useFoldAPI = kEmitFoldAdaptorFolder; 
    let cppNamespace = "::mlir::romaine";
}

//===----------------------------------------------------------------------===//
// Base romaine operation definition.
//===----------------------------------------------------------------------===//

class Romaine_Op<string mnemonic, list<Trait> traits = []> :
        Op<Romaine_Dialect, mnemonic, traits>;

#endif // ROMAINE_DIALECT
