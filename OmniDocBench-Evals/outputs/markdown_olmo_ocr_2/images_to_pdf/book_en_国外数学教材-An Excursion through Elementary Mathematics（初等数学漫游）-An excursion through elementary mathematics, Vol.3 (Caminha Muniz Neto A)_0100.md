\[
\frac{1 - 2x + 2x^2}{(1-x)^2(1-2x)} = \frac{A}{(1-x)^2} + \frac{B}{1-x} + \frac{C}{1-2x}.
\]

(d) Expand each summand of the right hand side above as a power series to conclude that \(a_n = 2^{n+1} - (n+1)\) for \(n \geq 0\).

3. Given \(k, m \in \mathbb{N}\), use generating functions to compute the number of integer solutions of the equation \(a_1 + a_2 + \cdots + a_k = m\), such that \(a_i \geq 1\) for \(1 \leq i \leq k\).
4. Use generating functions to compute the number of nonnegative integer solutions of the equation \(a_1 + a_2 + a_3 + a_4 = 20\), satisfying \(a_1 \geq 2\) and \(a_3 \leq 7\).
5. A particle moves on the cartesian plane in such a way that from point \((a, b)\) it can go to either \((a+1, b)\) or \((a, b+1)\). Given \(n \in \mathbb{N}\), let \(a_n\) be the number of distinct ways the particle has to go from \(A_0(0, 0)\) to \(A_n(n, n)\), without ever touching a point \((x, y)\) situated above the bisector of odd quadrants (i.e., one such point for which \(y > x\)). In this respect, do the following items:

(a) Let \(A_k(k, k)\), with \(0 \leq k < n\). Prove that there are exactly \(a_k a_{n-1-k}\) distinct trajectories for the particle in which \(A_k\) is the last point (before \(A_n\)) on the line \(y = x\).
(b) Conclude that \(a_n = \sum_{k=0}^{n-1} a_k a_{n-1-k}\) and, hence, that \(a_n = C_n\) for \(n \geq 1\), where \(C_n\) is the \(n\)-th Catalan number.

For the coming problem, the reader may find it convenient to read again the paragraph that precedes Example 1.15.

6. For \(n \in \mathbb{N}\), we let \(a_n\) denote the number of partitions of \(n\) in natural summands, none of which exceeds 3. The purpose of this problem is to compute \(a_n\) as a function of \(n\), and to this end do the following items:

(a) Show that, for \(|x| < 1\), one has

\[
\sum_{n \geq 1} a_n x^n = \frac{1}{(1-x)(1-x^2)(1-x^3)}.
\]

(b) Find \(a, b, c, d \in \mathbb{R}\) for which

\[
\frac{1}{(1-x)(1-x^2)(1-x^3)} = \frac{a}{(1-x)^3} + \frac{b}{(1-x)^2} + \frac{c}{1-x^2} + \frac{d}{1-x^3}.
\]

(c) Conclude that

\[
a_n = \begin{cases}
\frac{1}{6} \binom{n+2}{2} + \frac{1}{4}(n+1) + \frac{7}{12}, & \text{if } 6 \mid n \\
\frac{1}{6} \binom{n+2}{2} + \frac{1}{4}(n+1) + \frac{1}{4}, & \text{if } 2 \mid n \text{ but } 3 \nmid n \\
\frac{1}{6} \binom{n+2}{2} + \frac{1}{4}(n+1) + \frac{1}{3}, & \text{if } 2 \nmid n \text{ but } 3 \mid n \\
\frac{1}{6} \binom{n+2}{2} + \frac{1}{4}(n+1), & \text{if } 2, 3 \nmid n
\end{cases}.
\]