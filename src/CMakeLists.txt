set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/lexer.cpp
        ${CMAKE_CURRENT_LIST_DIR}/AST.cpp
        ${CMAKE_CURRENT_LIST_DIR}/ASTPrinter.cpp
        ${CMAKE_CURRENT_LIST_DIR}/parser.cpp
        )

set(HPP_FILES
        ${CMAKE_CURRENT_LIST_DIR}/include/lexer.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/AST.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/ASTVisitor.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/parser.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/ASTPrinter.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/AST.fwd.hpp
        )

set(ALL_SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/mlidk.cpp
        ${SRC_FILES}
        )
