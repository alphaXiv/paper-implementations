by comparing the sum to an integral. This gives

\[
n^{n^2/2-C_1 n} e^{-n^2/4} \leq 1^{1+c} 2^{2+c} \cdots n^{n+c}
\]
\[
\leq n^{n^2/2+C_2 n} e^{-n^2/4}.
\]

We now interpret \( f(n) \) as counting the number of \( n \)-tuples \((a_1, \ldots, a_n)\) of nonnegative integers such that

\[
a_1 1! + \cdots + a_n n! = n!.
\]

For an upper bound on \( f(n) \), we use the inequalities \( 0 \leq a_i \leq n!/i! \) to deduce that there are at most \( n!/i! + 1 \leq 2(n!/i!) \) choices for \( a_i \). Hence

\[
f(n) \leq 2^n \frac{n!}{1!} \cdots \frac{n!}{n!}
= 2^n 1^2 3^2 \cdots n^{n-1}
\leq n^{n^2/2+C_3 n} e^{-n^2/4}.
\]

For a lower bound on \( f(n) \), we note that if \( 0 \leq a_i < (n-1)!/i! \) for \( i = 2, \ldots, n-1 \) and \( a_n = 0 \), then \( 0 \leq a_2 2! + \cdots + a_{n-1} (n-1)! \leq n! \), so there is a unique choice of \( a_1 \) to complete this to a solution of \( a_1 1! + \cdots + a_n n! = n! \). Hence

\[
f(n) \geq \frac{(n-1)!}{2!} \cdots \frac{(n-1)!}{(n-1)!}
= 3^1 4^2 \cdots (n-1)^{n-3}
\geq n^{n^2/2+C_4 n} e^{-n^2/4}.
\]