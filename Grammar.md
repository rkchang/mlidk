# Grammar for MLIDK

```
prog ::= expr

expr ::= 'let' name '=' expr 'in' expr
       | 'if' expr 'then' expr 'else' expr
       | logic
       | func_def
       | func_call

func_def ::= 'fun' '(' [ ident ( ',' ident )* ] ')' expr
    
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
```
