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
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
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
