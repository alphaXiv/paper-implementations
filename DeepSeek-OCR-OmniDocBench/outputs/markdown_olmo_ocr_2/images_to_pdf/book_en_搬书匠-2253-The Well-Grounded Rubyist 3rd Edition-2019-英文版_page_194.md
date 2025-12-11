>> end
=> nil

Entire if statement evaluates to nil because it fails

Conditional statements interact with other aspects of Ruby syntax in a couple of ways that you need to be aware of—in particular, with assignment syntax. It’s worth looking in some detail at how conditionals behave in assignments, because it involves some interesting points about how Ruby parses code.

6.1.2 Assignment syntax in condition bodies and tests

Assignment syntax and conditional expressions cross paths at two points: in the bodies of conditional expressions, where the assignments may or may not happen at all, and in the conditional tests themselves:

if x = 1    ← Assignment in conditional
  y = 2    ← Assignment in conditional
end

What happens (or doesn’t) when you use these idioms? We’ll look at both, starting with variable assignment in the body of the conditional—specifically, local-variable assignment, which displays some perhaps unexpected behavior in this context.

LOCAL-VARIABLE ASSIGNMENT IN A CONDITIONAL BODY
Ruby doesn’t draw as clear a line as compiled languages do between “compile time” and “runtime,” but the interpreter does parse your code before running it, and certain decisions are made during that process. An important one is the recognition and allocation of local variables.

When the Ruby parser sees the sequence identifier, equal-sign, and value, as in this expression,

x = 1

it allocates space for a local variable called x. The creation of the variable—not the assignment of a value to it, but the internal creation of a variable—always takes place as a result of this kind of expression, even if the code isn’t executed!

Consider this example:

if false
  x = 1    Output: nil
end
p x        NameError: undefined local variable or method ‘y’
p y

The assignment to x isn’t executed, because it’s wrapped in a failing conditional test. But the Ruby parser sees the sequence x = 1, from which it deduces that the program involves a local variable x. The parser doesn’t care whether x is ever assigned a value. Its job is just to scour the code for local variables for which space needs to be allocated.

The result is that x inhabits a strange kind of variable limbo. It has been brought into being and initialized to nil. In that respect, it differs from a variable that has no existence at all; as you can see in the example, examining x gives you the value nil,