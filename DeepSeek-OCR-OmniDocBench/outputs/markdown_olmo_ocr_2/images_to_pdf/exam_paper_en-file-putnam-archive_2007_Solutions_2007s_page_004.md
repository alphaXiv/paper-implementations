B-2 Put \( B = \max_{0 \leq x \leq 1} |f'(x)| \) and \( g(x) = \int_0^x f(y) dy \). Since \( g(0) = g(1) = 0 \), the maximum value of \( |g(x)| \) must occur at a critical point \( y \in (0, 1) \) satisfying \( g''(y) = f(y) = 0 \). We may thus take \( \alpha = y \) hereafter.

Since \( \int_0^\alpha f(x) dx = -\int_0^{1-\alpha} f(1-x) dx \), we may assume that \( \alpha \leq 1/2 \). By then substituting \( -f(x) \) for \( f(x) \) if needed, we may assume that \( \int_0^\alpha f(x) dx \geq 0 \). From the inequality \( f'(x) \geq -B \), we deduce \( f(x) \leq B(\alpha - x) \) for \( 0 \leq x \leq \alpha \), so

\[
\int_0^\alpha f(x) dx \leq \int_0^\alpha B(\alpha - x) dx \\
= -\frac{1}{2} B(\alpha - x)^2 \Big|_0^\alpha \\
= \frac{\alpha^2}{2} B \leq \frac{1}{8} B
\]

as desired.

B-3 First solution: Observing that \( x_2/2 = 13, x_3/4 = 34, x_4/8 = 89 \), we guess that \( x_n = 2^n F_{2n+3} \), where \( F_k \) is the \( k \)-th Fibonacci number. Thus we claim that \( x_n = \frac{2^{n-1}}{\sqrt{5}} (\alpha^{2n+3} - \alpha^{-(2n+3)}) \), where \( \alpha = \frac{1+\sqrt{5}}{2} \), to make the answer \( x_{2007} = \frac{2^{2006}}{\sqrt{5}} (\alpha^{3997} - \alpha^{-3997}) \).

We prove the claim by induction; the base case \( x_0 = 1 \) is true, and so it suffices to show that the recursion \( x_{n+1} = 3x_n + \lfloor x_n \sqrt{5} \rfloor \) is satisfied for our formula for \( x_n \). Indeed, since \( \alpha^2 = \frac{3+\sqrt{5}}{2} \), we have

\[
x_{n+1} - (3+\sqrt{5})x_n = \frac{2^{n-1}}{\sqrt{5}} (2(\alpha^{2n+5} - \alpha^{-(2n+5)})) \\
- (3+\sqrt{5})(\alpha^{2n+3} - \alpha^{-(2n+3)})
= 2^n \alpha^{-(2n+3)}.
\]

Now \( 2^n \alpha^{-(2n+3)} = (1-\frac{\sqrt{5}}{2})^3 (3-\sqrt{5})^n \) is between \( -1 \) and 0; the recursion follows since \( x_n, x_{n+1} \) are integers.

Second solution: (by Catalin Zara) Since \( x_n \) is rational, we have \( 0 < x_n \sqrt{5} - \lfloor x_n \sqrt{5} \rfloor < 1 \). We now have the inequalities

\[
x_{n+1} - 3x_n < x_n \sqrt{5} < x_{n+1} - 3x_n + 1 \\
(3+\sqrt{5})x_n - 1 < x_{n+1} < (3+\sqrt{5})x_n \\
4x_n - (3-\sqrt{5}) < (3-\sqrt{5})x_{n+1} < 4x_n \\
3x_{n+1} - 4x_n < x_{n+1} \sqrt{5} < 3x_{n+1} - 4x_n + (3-\sqrt{5}).
\]

Since \( 0 < 3-\sqrt{5} < 1 \), this yields \( \lfloor x_{n+1} \sqrt{5} \rfloor = 3x_{n+1} - 4x_n \), so we can rewrite the recursion as \( x_{n+1} = 6x_n - 4x_{n-1} \) for \( n \geq 2 \). It is routine to solve this recursion to obtain the same solution as above.

Remark: With an initial 1 prepended, this becomes sequence A018903 in Sloane's On-Line Encyclopedia of Integer Sequences: (http://www.research.att.com/~njas/sequences/). Therein, the sequence is described as the case S(1, 5) of the sequence S(a_0, a_1) in which \( a_{n+2} \) is the least integer for which \( a_{n+2}/a_{n+1} > a_{n+1}/a_n \). Sloane cites D. W. Boyd, Linear recurrence relations for some generalized Pisot sequences, Advances in Number Theory (Kingston, ON, 1991), Oxford Univ. Press, New York, 1993, p. 333–340.

B-4 The number of pairs is \( 2^{n+1} \). The degree condition forces P to have degree n and leading coefficient ±1; we may count pairs in which P has leading coefficient 1 as long as we multiply by 2 afterward.

Factor both sides:

\[
(P(X) + Q(X)i)(P(X) - Q(X)i) \\
= \prod_{j=0}^{n-1} (X - \exp(2\pi i (2j+1)/(4n))) \\
\cdot \prod_{j=0}^{n-1} (X + \exp(2\pi i (2j+1)/(4n))).
\]

Then each choice of \( P, Q \) corresponds to equating \( P(X) + Q(X)i \) with the product of some n factors on the right, in which we choose exactly of the two factors for each \( j = 0, \ldots, n-1 \). (We must take exactly n factors because as a polynomial in X with complex coefficients, \( P(X) + Q(X)i \) has degree exactly n. We must choose one for each j to ensure that \( P(X) + Q(X)i \) and \( P(X) - Q(X)i \) are complex conjugates, so that \( P, Q \) have real coefficients.) Thus there are \( 2^n \) such pairs; multiplying by 2 to allow P to have leading coefficient −1 yields the desired result.

Remark: If we allow P and Q to have complex coefficients but still require \( \deg(P) > \deg(Q) \), then the number of pairs increases to \( 2\binom{2n}{n} \), as we may choose any n of the 2n factors of \( X^{2n} + 1 \) to use to form \( P(X) + Q(X)i \).

B-5 For n an integer, we have \( \left\lfloor \frac{n}{k} \right\rfloor = \frac{n-j}{k} \) for j the unique integer in \( \{0, \ldots, k-1\} \) congruent to n modulo k; hence

\[
\prod_{j=0}^{k-1} \left( \left\lfloor \frac{n}{k} \right\rfloor - \frac{n-j}{k} \right) = 0.
\]

By expanding this out, we obtain the desired polynomials \( P_0(n), \ldots, P_{k-1}(n) \).

Remark: Variants of this solution are possible that construct the \( P_i \) less explicitly, using Lagrange interpolation or Vandermonde determinants.

B-6 (Suggested by Oleg Golberg) Assume \( n \geq 2 \), or else the problem is trivially false. Throughout this proof, any \( C_i \) will be a positive constant whose exact value is immaterial. As in the proof of Stirling’s approximation, we estimate for any fixed \( c \in \mathbb{R} \),

\[
\sum_{i=1}^n (i+c) \log i = \frac{1}{2} n^2 \log n - \frac{1}{4} n^2 + O(n \log n)
\]