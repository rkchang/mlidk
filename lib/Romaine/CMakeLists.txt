add_mlir_dialect_library(MLIRRomaine
        RomaineDialect.cpp
        RomaineOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Romaine

        DEPENDS
        MLIRRomaineOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
	)
message("lib/romaine")