A–5 It suffices to prove that for any relatively prime positive integers \( r, s \), there exists an integer \( n \) with \( a_n = r \) and \( a_{n+1} = s \). We prove this by induction on \( r + s \), the case \( r + s = 2 \) following from the fact that \( a_0 = a_1 = 1 \). Given \( r \) and \( s \) not both 1 with \( \gcd(r, s) = 1 \), we must have \( r \neq s \). If \( r > s \), then by the induction hypothesis we have \( a_n = r - s \) and \( a_{n+1} = s \) for some \( n \); then \( a_{2n+2} = r \) and \( a_{2n+3} = s \). If \( r < s \), then we have \( a_n = r \) and \( a_{n+1} = s - r \) for some \( n \); then \( a_{2n+1} = r \) and \( a_{2n+2} = s \).

Note: a related problem is as follows. Starting with the sequence

\[
\begin{array}{c}
0 \\
1 \\
1 \\
0
\end{array}
\]

repeat the following operation: insert between each pair \( \frac{a}{b} \) and \( \frac{c}{d} \) the pair \( \frac{a+c}{b+d} \). Prove that each positive rational number eventually appears.

Observe that by induction, if \( \frac{a}{b} \) and \( \frac{c}{d} \) are consecutive terms in the sequence, then \( bc - ad = 1 \). The same holds for consecutive terms of the \( n \)-th Farey sequence, the sequence of rational numbers in \([0, 1]\) with denominator (in lowest terms) at most \( n \).

A–6 The sum converges for \( b = 2 \) and diverges for \( b \geq 3 \). We first consider \( b \geq 3 \). Suppose the sum converges; then the fact that \( f(n) = nf(d) \) whenever \( b^{d-1} \leq n \leq b^d - 1 \) yields

\[
\sum_{n=1}^{\infty} \frac{1}{f(n)} = \sum_{d=1}^{\infty} \frac{1}{f(d)} \sum_{n=b^{d-1}}^{b^d-1} \frac{1}{n}.
\] (1)

However, by comparing the integral of \( 1/x \) with a Riemann sum, we see that

\[
\sum_{n=b^{d-1}}^{b^d-1} \frac{1}{n} > \int_{b^{d-1}}^{b^d} \frac{dx}{x} = \log(b^d) - \log(b^{d-1}) = \log b,
\]

where \( \log \) denotes the natural logarithm. Thus (1) yields

\[
\sum_{n=1}^{\infty} \frac{1}{f(n)} > (\log b) \sum_{n=1}^{\infty} \frac{1}{f(n)},
\]

a contradiction since \( \log b > 1 \) for \( b \geq 3 \). Therefore the sum diverges.

For \( b = 2 \), we have a slightly different identity because \( f(2) \neq 2f(2) \). Instead, for any positive integer \( i \), we have

\[
\sum_{n=1}^{2^i-1} \frac{1}{f(n)} = 1 + \frac{1}{2} + \frac{1}{6} + \sum_{d=3}^{i} \frac{1}{f(d)} \sum_{n=2^{d-1}}^{2^d-1} \frac{1}{n}.
\] (2)

Again comparing an integral to a Riemann sum, we see that for \( d \geq 3 \),

\[
\sum_{n=2^{d-1}}^{2^d-1} \frac{1}{n} < \frac{1}{2^{d-1}} - \frac{1}{2^d} + \int_{2^{d-1}}^{2^d} \frac{dx}{x}
\]
\[
= \frac{1}{2^d} + \log 2
\]
\[
\leq \frac{1}{8} + \log 2 < 0.125 + 0.7 < 1.
\]

Put \( c = \frac{1}{8} + \log 2 \) and \( L = 1 + \frac{1}{2} + \frac{1}{6(1-c)} \). Then we can prove that \( \sum_{n=1}^{2^i-1} \frac{1}{f(n)} < L \) for all \( i \geq 2 \) by induction on \( i \). The case \( i = 2 \) is clear. For the induction, note that by (2),

\[
\sum_{n=1}^{2^i-1} \frac{1}{f(n)} < 1 + \frac{1}{2} + \frac{1}{6} + c \sum_{d=3}^{i} \frac{1}{f(d)}
\]
\[
< 1 + \frac{1}{2} + \frac{1}{6} + c \frac{1}{6(1-c)}
\]
\[
= 1 + \frac{1}{2} + \frac{1}{6(1-c)} = L,
\]

as desired. We conclude that \( \sum_{n=1}^{\infty} \frac{1}{f(n)} \) converges to a limit less than or equal to \( L \).

Note: the above argument proves that the sum for \( b = 2 \) is at most \( L < 2.417 \). One can also obtain a lower bound by the same technique, namely \( 1 + \frac{1}{2} + \frac{1}{6(1-c')} \) with \( c' = \log 2 \). This bound exceeds 2.043. (By contrast, summing the first 100000 terms of the series only yields a lower bound of 1.906.) Repeating the same arguments with \( d \geq 4 \) as the cutoff yields the upper bound 2.185 and the lower bound 2.079.

B–1 The probability is 1/99. In fact, we show by induction on \( n \) that after \( n \) shots, the probability of having made any number of shots from 1 to \( n-1 \) is equal to \( 1/(n-1) \). This is evident for \( n = 2 \). Given the result for \( n \), we see that the probability of making \( i \) shots after \( n+1 \) attempts is

\[
\frac{i-1}{n} \cdot \frac{1}{n-1} + \left(1 - \frac{i}{n}\right) \frac{1}{n-1} = \frac{(i-1)+(n-i)}{n(n-1)} = \frac{1}{n},
\]

as claimed.

B–2 (Note: the problem statement assumes that all polyhedra are connected and that no two edges share more than one face, so we will do likewise. In particular, these are true for all convex polyhedra.) We show that in fact the first player can win on the third move. Suppose the polyhedron has a face \( A \) with at least four edges. If the first player plays there first, after the second player's first move there will be three consecutive faces \( B, C, D \) adjacent to \( A \) which are all unoccupied. The first player wins by playing in \( C \); after the second player's second