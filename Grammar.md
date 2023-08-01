# Grammar for MLIDK

```
prog ::= expr

expr ::= 'let' defns 'in' expr
       | 'if' expr 'then' expr 'else' expr
       | '|' params '|' expr
       | logic

defns ::= ( 'def' def_binder )+
        | let_binder

let_binder = name '=' expr
def_binder = name '(' params ')' '->' type '=' expr

params ::= [ param ( ',' param )* ]
param  ::= ident ':' type

logic ::= equality ( ( 'and' | 'or' ) equality )*

equality ::= comparisson ( ( '==' | '!=' ) comparisson )*

comparisson ::= term ( ( '<' | '<=' | '>' | '>=' ) term )*

term ::= factor ( ( '+' | '-' ) factor )*

factor ::= unary ( ( '*' | '/' ) unary )*

unary ::= [ 'not' ] func_call

func_call ::= primary ( '(' [ expr ( ',' expr )* ] ')' )*

primary ::= '(' expr ')' | int | name | bool

bool ::= 'true' | 'false'
int  ::= [0-9]+
name ::= [a-z][a-zA-Z0-9_]*

type ::= '(' [ type ( ',' type )* ] ')' '->' type
       | 'i32'
       | 'bool'
       | 'void'
```
