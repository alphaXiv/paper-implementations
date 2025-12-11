product of the orders of all of its elements cannot be a power of 2.

We may thus consider only abelian 2-groups hereafter. For such a group G, the product of the orders of all of its elements has the form \(2^{k(G)}\) for some nonnegative integer \(G\), and we must show that it is impossible to achieve \(k(G) = 2009\). Again by the structure theorem, we may write

\[
G \cong \prod_{i=1}^{\infty} (\mathbb{Z}/2^{e_i}\mathbb{Z})^{e_i}
\]

for some nonnegative integers \(e_1, e_2, \ldots\), all but finitely many of which are 0.

For any nonnegative integer \(m\), the elements of \(G\) of order at most \(2^m\) form a subgroup isomorphic to

\[
\prod_{i=1}^{\infty} (\mathbb{Z}/2^{\min\{i,m\}}\mathbb{Z})^{e_i},
\]

which has \(2^{s_m}\) elements for \(s_m = \sum_{i=1}^{\infty} \min\{i,m\} e_i\). Hence

\[
k(G) = \sum_{i=1}^{\infty} i(2^{s_i} - 2^{s_{i-1}}).
\]

Since \(s_1 \leq s_2 \leq \cdots\), \(k(G) + 1\) is always divisible by \(2^{s_1}\). In particular, \(k(G) = 2009\) forces \(s_1 \leq 1\).

However, the only cases where \(s_1 \leq 1\) are where all of the \(e_i\) are 0, in which case \(k(G) = 0\), or where \(e_i = 1\) for some \(i\) and \(e_j = 0\) for \(j \neq i\), in which case \(k(G) = (i-1)2^i + 1\). The right side is a strictly increasing function of \(i\) which equals 1793 for \(i = 8\) and 4097 for \(i = 9\), so it can never equal 2009. This proves the claim.

Remark. One can also arrive at the key congruence by dividing \(G\) into equivalence classes, by declaring two elements to be equivalent if they generate the same cyclic subgroup of \(G\). For \(h > 0\), an element of order \(2^h\) belongs to an equivalence class of size \(2^{h-1}\), so the products of the orders of the elements of this equivalence class is \(2^j\) for \(j = h2^{h-1}\). This quantity is divisible by 4 as long as \(h > 1\); thus to have \(k(G) \equiv 1 \pmod{4}\), the number of elements of \(G\) of order 2 must be congruent to 1 modulo 4. However, there are exactly \(2^e - 1\) such elements, for \(e\) the number of cyclic factors of \(G\). Hence \(e = 1\), and one concludes as in the given solution.

A–6 We disprove the assertion using the example

\[
f(x, y) = 3(1 + y)(2x - 1)^2 - y.
\]

We have \(b - a = d - c = 0\) because the identity \(f(x, y) = f(1 - x, y)\) forces \(a = b\), and because

\[
c = \int_0^1 3(2x - 1)^2 dx = 1,
\]
\[
d = \int_0^1 (6(2x - 1)^2 - 1) dx = 1.
\]

Moreover, the partial derivatives

\[
\frac{\partial f}{\partial x}(x_0, y_0) = 3(1 + y_0)(8x_0 - 4)
\]
\[
\frac{\partial f}{\partial y}(x_0, y_0) = 3(2x_0 - 1)^2 - 1.
\]

have no common zero in \((0, 1)^2\). Namely, for the first partial to vanish, we must have \(x_0 = 1/2\) since \(1 + y_0\) is nowhere zero, but for \(x_0 = 1/2\) the second partial cannot vanish.

Remark. This problem amounts to refuting a potential generalization of the Mean Value Theorem to bivariate functions. Many counterexamples are possible. Kent Merryfield suggests \(y \sin(2\pi x)\), for which all four of the boundary integrals vanish; here the partial derivatives are \(2\pi y \cos(2\pi x)\) and \(\sin(2\pi x)\). Catalin Zara suggests \(x^{1/3} y^{2/3}\). Qingchun Ren suggests \(xy(1 - y)\).

B–1 Every positive rational number can be uniquely written in lowest terms as \(a/b\) for \(a, b\) positive integers. We prove the statement in the problem by induction on the largest prime dividing either \(a\) or \(b\) (where this is considered to be 1 if \(a = b = 1\)). For the base case, we can write \(1/1 = 2! / 2!\). For a general \(a/b\), let \(p\) be the largest prime dividing either \(a\) or \(b\); then \(a/b = p^k a'/b'\) for some \(k \neq 0\) and positive integers \(a', b'\) whose largest prime factors are strictly less than \(p\). We now have \(a/b = (p!)^k \frac{a'}{(p-1)!^{k'} b'}\), and all prime factors of \(a'\) and \((p-1)!^{k'} b'\) are strictly less than \(p\). By the induction assumption, \(\frac{a'}{(p-1)!^{k'} b'}\) can be written as a quotient of products of prime factorials, and so \(a/b = (p!)^k \frac{a'}{(p-1)!^{k'} b'}\) can as well. This completes the induction.

Remark. Noam Elkies points out that the representations are unique up to rearranging and canceling common factors.

B–2 The desired real numbers \(c\) are precisely those for which \(1/3 < c \leq 1\). For any positive integer \(m\) and any sequence \(0 = x_0 < x_1 < \cdots < x_m = 1\), the cost of jumping along this sequence is \(\sum_{i=1}^m (x_i - x_{i-1}) x_i^2\). Since

\[
1 = \sum_{i=1}^m (x_i - x_{i-1}) \geq \sum_{i=1}^m (x_i - x_{i-1}) x_i^2
\]
\[
> \sum_{i=1}^m \int_{x_{i-1}}^{x_i} t^2 dt
\]
\[
= \int_0^1 t^2 dt = \frac{1}{3},
\]

we can only achieve costs \(c\) for which \(1/3 < c \leq 1\).

It remains to check that any such \(c\) can be achieved. Suppose \(0 = x_0 < \cdots < x_m = 1\) is a sequence with \(m \geq 1\). For \(i = 1, \ldots, m\), let \(c_i\) be the cost of the sequence \(0, x_i, x_{i+1}, \ldots, x_m\). For \(i > 1\) and \(0 < y \leq x_{i-1}\), the cost of the sequence \(0, y, x_i, \ldots, x_m\) is

\[
c_i + y^3 + (x_i - y)x_i^2 - x_i^3 = c_i - y(x_i^2 - y^2),
\]