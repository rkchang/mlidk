enable_testing()

# googletest target for all unittests
add_executable(all_tests
        lexer_test.cpp
        parser_test.cpp
        ${PROJECT_SOURCE_DIR}/lettucec/AST.cpp
        ${PROJECT_SOURCE_DIR}/lettucec/types.cpp
        ${PROJECT_SOURCE_DIR}/lettucec/type_checker.cpp
        ${PROJECT_SOURCE_DIR}/lettucec/lexer.cpp
        ${PROJECT_SOURCE_DIR}/lettucec/ASTPrinter.cpp
        ${PROJECT_SOURCE_DIR}/lettucec/parser.cpp
        )

target_compile_options(all_tests PRIVATE
        -Wall
        -Werror
        -Wpedantic
        -Wextra
        -fPIE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -fno-optimize-sibling-calls
        -g
      )

target_link_options(all_tests PRIVATE
        -fsanitize=address,undefined
        -g
      )

target_include_directories(all_tests PRIVATE "${PROJECT_SOURCE_DIR}/include")

target_link_libraries(all_tests
        GTest::gtest_main
        ${LIBS}
        )


include(GoogleTest)
gtest_discover_tests(all_tests)

