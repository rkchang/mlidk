
# Grammar for MLIDK

```
prog ::= expr

expr ::= 'let' name '=' expr 'in' expr
       | term ( ( '+' | '-' ) term )*

term ::= factor ( ( '*' | '/' ) factor )*

factor ::= '(' expr ')' | int | name

int  ::= [0-9]+
name ::= [a-z][a-zA-Z0-9_]*
```