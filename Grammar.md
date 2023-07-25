# Grammar for MLIDK

```
prog ::= expr

expr ::= 'let' name '=' expr 'in' expr
       | 'if' expr 'then' expr 'else' expr
       | '\' param ( ',' param )* '.' expr
       | logic
       | func_call

param ::= ident ':' type

func_call ::= ident '(' [ expr ( ',' expr )* ] ')'

logic ::= equality ( ( 'and' | 'or' ) equality )*

equality ::= comparisson ( ( '==' | '!=' ) comparisson )*

comparisson ::= term ( ( '<' | '<=' | '>' | '>=' ) term )*

term ::= factor ( ( '+' | '-' ) factor )*

factor ::= unary ( ( '*' | '/' ) unary )*

unary ::= [ 'not' ] primary

primary ::= '(' expr ')' | int | name | bool

bool ::= 'true' | 'false'
int  ::= [0-9]+
name ::= [a-z][a-zA-Z0-9_]*

type ::= '(' [ type ( ',' type )* ] ')' '->' type
       | 'i32'
       | 'bool'
       | 'void'
```
