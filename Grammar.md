# Grammar for MLIDK

```
prog ::= expr

expr ::= 'let' name '=' expr 'in' expr
         | term
       
term ::= factor ( ( '+' | '-' ) factor )*

factor ::= primary ( ( '*' | '/' ) primary )*

primary ::= '(' expr ')' | int | name

int  ::= [0-9]+
name ::= [a-z][a-zA-Z0-9_]*
```