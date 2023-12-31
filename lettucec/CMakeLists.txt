set(CMAKE_CXX_CLANG_TIDY clang-tidy -extra-arg=-Wno-unknown-warning-option)

add_executable(lettucec
        lettucec.cpp # main file for letttucec
        types.cpp
        AST.cpp
        type_checker.cpp
        ASTPrinter.cpp
        lexer.cpp
        parser.cpp
        passes.cpp
        MLIR_gen.cpp)


# Debug Build flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
  -fsanitize=address \
  -fno-omit-frame-pointer \
  -fsanitize=undefined \
  -fno-optimize-sibling-calls \
  -g"
)
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} \
  -fsanitize=address,undefined \
  -g"
)

# General Build flags
target_compile_options(lettucec PRIVATE
        -Wall
        -Wpedantic
        -Wextra
        -fPIE
        -Werror
)

llvm_map_components_to_libnames(llvm_libs support core irreader orcjit nativecodegen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        ${llvm_libs}
        MLIROptLib
        MLIRLLVMToLLVMIRTranslation
        MLIRExecutionEngine
)


# Link against LLVM and MLIR libraries
target_link_libraries(lettucec ${LIBS})

mlir_check_all_link_libraries(lettucec)

