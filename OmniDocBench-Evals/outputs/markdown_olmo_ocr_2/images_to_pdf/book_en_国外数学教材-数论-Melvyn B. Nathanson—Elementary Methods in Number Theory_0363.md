Hint: Observe that if \( k \geq 1 \) and \( f(x) \) is integer-valued for all sufficiently large \( x \), then \( \Delta f(x) \) is also integer-valued for all sufficiently large \( x \). Represent \( f(x) \) in the form (11.1) and use induction on \( k \).

5. Let \( f(x) \) be a polynomial of degree \( k \) with complex coefficients. Prove that if \( f(x) \) is an integer for all sufficiently large integers \( x \), then \( f(x) \) is an integer for all integers \( x \).

6. Prove that if \( f(x) \) is an integer-valued polynomial of degree \( k \) with leading coefficient \( a_k \), then

\[
|a_k| \geq \frac{1}{k!}.
\]

7. Let \( f(x) \) be an integer-valued polynomial, and define

\[
d = \gcd\{f(x) : x \in \mathbf{N}_0\}
\]

and

\[
d' = \gcd\{f(x) : x \in \mathbf{Z}\}.
\]

Let \( u_0, u_1, \ldots, u_k \) be integers such that

\[
f(x) = \sum_{i=0}^k u_i \binom{x}{i}.
\]

Prove that

\[
d = d' = (u_0, u_1, \ldots, u_k).
\]

8. Prove that if

\[
f(x) = \sum_{i=0}^k u_i \binom{x}{i},
\]

then

\[
f_1(x) = f(x+1) = u_k \binom{x}{k} + \sum_{i=0}^{k-1} (u_i + u_{i+1}) \binom{x}{i}.
\]

Prove that

\[
\begin{align*}
&\gcd(u_0, u_1, \ldots, u_{k-1}, u_k) \\
&\quad= \gcd(u_0 + u_1, u_1 + u_2, \ldots, u_{k-1} + u_k, u_k).
\end{align*}
\]

9. Let \( f(x) \) be an integer-valued polynomial and let \( m \in \mathbf{Z} \). We define the polynomial \( f_m(x) = f(x+m) \). Prove that \( f(x) \) and \( f_m(x) \) are polynomials of the same degree and with the same leading coefficient. Let \( A(f) = \{f(i)\}_{i=0}^\infty \). Prove that \( \gcd(A(f)) = \gcd(A(f_m)) \).