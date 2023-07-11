add_executable(lettucec
        mlidk.cpp # main file for letttucec
        AST.cpp
        ASTPrinter.cpp
        lexer.cpp
        parser.cpp
)

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
        -Werror
        -Wpedantic
        -Wextra
        -fPIE
        )


# Link against LLVM and MLIR libraries
target_link_libraries(lettucec ${LIBS} )

mlir_check_all_link_libraries(lettucec)

