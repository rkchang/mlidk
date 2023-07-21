# Grammar for MLIDK

```
prog ::= expr

expr ::= 'let' name '=' expr 'in' expr
       | 'if' expr 'then' expr 'else' expr
       | term

term ::= factor ( ( '+' | '-' ) factor )*

factor ::= primary ( ( '*' | '/' ) primary )*

primary ::= '(' expr ')' | int | name | bool

bool ::= 'true' | 'false'
int  ::= [0-9]+
name ::= [a-z][a-zA-Z0-9_]*
```
