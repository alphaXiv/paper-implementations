Solutions to the 60th William Lowell Putnam Mathematical Competition
Saturday, December 4, 1999

Manjul Bhargava, Kiran Kedlaya, and Lenny Ng

A–1 Note that if \( r(x) \) and \( s(x) \) are any two functions, then
\[
\max(r,s) = (r + s + |r - s|)/2.
\]
Therefore, if \( F(x) \) is the given function, we have
\[
F(x) = \max\{-3x-3,0\} - \max\{5x,0\} + 3x + 2 \\
= (-3x-3 + |3x+3|)/2 \\
\quad - (5x + |5x|)/2 + 3x + 2 \\
= |(3x+3)/2| - |5x/2| - x + \frac{1}{2},
\]
so we may set \( f(x) = (3x+3)/2,\ g(x) = 5x/2,\ ) and \( h(x) = -x + \frac{1}{2} \).

A–2 First solution: First factor \( p(x) = q(x)r(x) \), where \( q \) has all real roots and \( r \) has all complex roots. Notice that each root of \( q \) has even multiplicity, otherwise \( p \) would have a sign change at that root. Thus \( q(x) \) has a square root \( s(x) \).

Now write \( r(x) = \prod_{j=1}^k (x-a_j)(x-\overline{a_j}) \) (possible because \( r \) has roots in complex conjugate pairs). Write \( \prod_{j=1}^k (x-a_j) = t(x) + iu(x) \) with \( t,x \) having real coefficients. Then for \( x \) real,
\[
p(x) = q(x)r(x) \\
= s(x)^2(t(x) + iu(x))(t(x) + iu(x)) \\
= (s(x)t(x))^2 + (s(x)u(x))^2.
\]
(Alternatively, one can factor \( r(x) \) as a product of quadratic polynomials with real coefficients, write each as a sum of squares, then multiply together to get a sum of many squares.)

Second solution: We proceed by induction on the degree of \( p \), with base case where \( p \) has degree 0. As in the first solution, we may reduce to a smaller degree in case \( p \) has any real roots, so assume it has none. Then \( p(x) > 0 \) for all real \( x \), and since \( p(x) \to \infty \) for \( x \to \pm \infty \), \( p \) has a minimum value \( c \). Now \( p(x) - c \) has real roots, so as above, we deduce that \( p(x) - c \) is a sum of squares. Now add one more square, namely \( (\sqrt{c})^2 \), to get \( p(x) \) as a sum of squares.

A–3 First solution: Computing the coefficient of \( x^{n+1} \) in the identity \( (1-2x-x^2)\sum_{m=0}^\infty a_m x^m = 1 \) yields the recurrence \( a_{n+1} = 2a_n + a_{n-1} \); the sequence \( \{a_n\} \) is then characterized by this recurrence and the initial conditions \( a_0 = 1, a_1 = 2 \).

Define the sequence \( \{b_n\} \) by \( b_{2n} = a_{n-1}^2 + a_n^2,\ b_{2n+1} = a_n(a_{n-1} + a_{n+1}) \). Then
\[
2b_{2n+1} + b_{2n} = 2a_n a_{n+1} + 2a_{n-1} a_n + a_{n-1}^2 + a_n^2 \\
= 2a_n a_{n+1} + a_{n-1} a_{n+1} + a_n^2 \\
= a_{n+1}^2 + a_n^2 = b_{2n+2},
\]
and similarly \( 2b_{2n} + b_{2n-1} = b_{2n+1} \), so that \( \{b_n\} \) satisfies the same recurrence as \( \{a_n\} \). Since further \( b_0 = 1, b_1 = 2 \) (where we use the recurrence for \( \{a_n\} \) to calculate \( a_{-1} = 0 \)), we deduce that \( b_n = a_n \) for all \( n \). In particular, \( a_n^2 + a_{n+1}^2 = b_{2n+2} = a_{2n+2} \).

Second solution: Note that
\[
\frac{1}{1-2x-x^2} = \frac{1}{2\sqrt{2}} \left( \frac{\sqrt{2}+1}{1-(1+\sqrt{2})x} + \frac{\sqrt{2}-1}{1-(1-\sqrt{2})x} \right)
\]
and that
\[
\frac{1}{1+(1\pm\sqrt{2})x} = \sum_{n=0}^\infty (1\pm\sqrt{2})^n x^n,
\]
so that
\[
a_n = \frac{1}{2\sqrt{2}} \left( ((\sqrt{2}+1)^{n+1} - (1-\sqrt{2})^{n+1}) \right).
\]
A simple computation (omitted here) now shows that \( a_n^2 + a_{n+1}^2 = a_{2n+2} \).

Third solution (by Richard Stanley): Let \( A \) be the matrix \( \begin{pmatrix} 0 & 1 \\ 1 & 2 \end{pmatrix} \). A simple induction argument shows that
\[
A^{n+2} = \begin{pmatrix} a_n & a_{n+1} \\ a_{n+1} & a_{n+2} \end{pmatrix}.
\]
The desired result now follows from comparing the top left corner entries of the equality \( A^{n+2}A^{n+2} = A^{2n+4} \).

A–4 Denote the series by \( S \), and let \( a_n = 3^n/n \). Note that
\[
S = \sum_{m=1}^\infty \sum_{n=1}^\infty \frac{1}{a_m(a_m + a_n)} \\
= \sum_{m=1}^\infty \sum_{n=1}^\infty \frac{1}{a_n(a_m + a_n)},
\]
where the second equality follows by interchanging \( m \)