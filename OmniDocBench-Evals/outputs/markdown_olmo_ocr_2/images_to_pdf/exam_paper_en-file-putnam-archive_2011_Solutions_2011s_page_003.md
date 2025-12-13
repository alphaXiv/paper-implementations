B1 Since the rational numbers are dense in the reals, we can find positive integers \(a, b\) such that
\[
\frac{3\varepsilon}{hk} < \frac{b}{a} < \frac{4\varepsilon}{hk}.
\]
By multiplying \(a\) and \(b\) by a suitably large positive integer, we can also ensure that \(3a^2 > b\). We then have
\[
\frac{\varepsilon}{hk} < \frac{b}{3a} < \frac{b}{\sqrt{a^2 + b + a}} = \sqrt{a^2 + b - a}
\]
and
\[
\sqrt{a^2 + b - a} = \frac{b}{\sqrt{a^2 + b + a}} \leq \frac{b}{2a} < \frac{2\varepsilon}{hk}.
\]
We may then take \(m = k^2(a^2 + b), n = h^2a^2\).

B2 Only the primes 2 and 5 appear seven or more times. The fact that these primes appear is demonstrated by the examples
\[(2, 5, 2), (2, 5, 3), (2, 7, 5), (2, 11, 5)\]
and their reversals.

It remains to show that if either \(\ell = 3\) or \(\ell\) is a prime greater than 5, then \(\ell\) occurs at most six times as an element of a triple in \(S\). Note that \((p, q, r) \in S\) if and only if \(q^2 - 4pr = a^2\) for some integer \(a\); in particular, since \(4pr \geq 16\), this forces \(q \geq 5\). In particular, \(q\) is odd, as then is \(a\), and so \(q^2 \equiv a^2 \equiv 1 \pmod{8}\); consequently, one of \(p, r\) must equal 2. If \(r = 2\), then \(8p = q^2 - a^2 = (q + a)(q - a)\); since both factors are of the same sign and their sum is the positive number \(2q\), both factors are positive. Since they are also both even, we have \(q + a \in \{2, 4, 2p, 4p\}\) and so \(q \in \{2p + 1, p + 2\}\). Similarly, if \(p = 2\), then \(q \in \{2r + 1, r + 2\}\). Consequently, \(\ell\) occurs at most twice as many times as there are prime numbers in the list
\[2\ell + 1, \ell + 2, \frac{\ell - 1}{2}, \ell - 2.\]
For \(\ell = 3, \ell - 2 = 1\) is not prime. For \(\ell \geq 7\), the numbers \(\ell - 2, \ell + 2\) cannot all be prime, since one of them is always a nontrivial multiple of 3.

Remark. The above argument shows that the cases listed for 5 are the only ones that can occur. By contrast, there are infinitely many cases where 2 occurs if either the twin prime conjecture holds or there are infinitely many Sophie Germain primes (both of which are expected to be true).

B3 Yes, it follows that \(f\) is differentiable.

First solution. Note first that at 0, \(f/g\) and \(g\) are both continuous, as then is their product \(f\). If \(f(0) \neq 0\), then in some neighborhood of 0, \(f\) is either always positive or always negative. We can thus choose \(\varepsilon \in \{\pm 1\}\) so that \(\varepsilon f\) is the composition of the differentiable function \((fg) \cdot (f/g)\) with the square root function. By the chain rule, \(f\) is differentiable at 0.

If \(f(0) = 0\), then \((f/g)(0) = 0\), so we have
\[
(f/g)'(0) = \lim_{x \to 0} \frac{f(x)}{xg(x)}.
\]
Since \(g\) is continuous at 0, we may multiply limits to deduce that \(\lim_{x \to 0} f(x)/x\) exists.

Second solution. Choose a neighborhood \(N\) of 0 on which \(g(x) \neq 0\). Define the following functions on \(N \setminus \{0\}\): \(h_1(x) = \frac{f(x)g(x) - f(0)g(0)}{x}\); \(h_2(x) = \frac{f(x)g(0) - f(0)g(x)}{xg(0)g(x)}\); \(h_3(x) = g(0)g(x)\); \(h_4(x) = \frac{1}{g(x)^2 + g(0)^2}\). Then by assumption, \(h_1, h_2, h_3, h_4\) all have limits as \(x \to 0\). On the other hand,
\[
\frac{f(x) - f(0)}{x} = (h_1(x) + h_2(x)h_3(x))h_4(x),
\]
and it follows that \(\lim_{x \to 0} \frac{f(x) - f(0)}{x}\) exists, as desired.

B4 Number the games 1, \ldots, 2011, and let \(A = (a_{jk})\) be the \(2011 \times 2011\) matrix whose \(jk\) entry is 1 if player \(k\) wins game \(j\) and \(i = \sqrt{-1}\) if player \(k\) loses game \(j\). Then \(\overline{a_{jk}}a_{jk}\) is 1 if players \(h\) and \(k\) tie in game \(j\); if player \(h\) wins and player \(k\) loses in game \(j\); and \(-i\) if \(h\) loses and \(k\) wins. It follows that \(T + iW = A^T A\).

Now the determinant of \(A\) is unchanged if we subtract the first row of \(A\) from each of the other rows, producing a matrix whose rows, besides the first one, are \((1 - i)\) times a row of integers. Thus we can write \(\det A = (1 - i)^{2010}(a + bi)\) for some integers \(a, b\). But then \(\det(T + iW) = \det(A^T A) = 2^{2010}(a^2 + b^2)\) is a non-negative integer multiple of \(2^{2010}\), as desired.

B5 Define the function
\[
f(y) = \int_{-\infty}^{\infty} \frac{dx}{(1 + x^2)(1 + (x + y)^2)}.
\]
For \(y \geq 0\), in the range \(-1 \leq x \leq 0\), we have
\[
(1 + x^2)(1 + (x + y)^2) \leq (1 + 1)(1 + (1 + y)^2) = 2y^2 + 4y + 4 \\
\leq 2y^2 + 4 + 2(y^2 + 1) \leq 6 + 6y^2.
\]
We thus have the lower bound
\[
f(y) \geq \frac{1}{6(1 + y^2)},
\]
the same bound is valid for \(y \leq 0\) because \(f(y) = f(-y)\).
The original hypothesis can be written as
\[
\sum_{i,j=1}^n f(a_i - a_j) \leq An
\]
and thus implies that
\[
\sum_{i,j=1}^n \frac{1}{1 + (a_i - a_j)^2} \leq 6An.
\]