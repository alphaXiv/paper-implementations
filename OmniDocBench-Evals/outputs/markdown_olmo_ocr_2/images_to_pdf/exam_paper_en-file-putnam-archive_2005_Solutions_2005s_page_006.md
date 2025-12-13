B–5 First solution: Put \( Q = x_1^2 + \cdots + x_n^2 \). Since \( Q \) is homogeneous, \( P \) is divisible by \( Q \) if and only if each of the homogeneous components of \( P \) is divisible by \( Q \). It is thus sufficient to solve the problem in case \( P \) itself is homogeneous, say of degree \( d \).

Suppose that we have a factorization \( P = Q^m R \) for some \( m > 0 \), where \( R \) is homogeneous of degree \( d \) and not divisible by \( Q \); note that the homogeneity implies that

\[
\sum_{i=1}^n x_i \frac{\partial R}{\partial x_i} = dR.
\]

Write \( \nabla^2 \) as shorthand for \( \frac{\partial^2}{\partial x_1^2} + \cdots + \frac{\partial^2}{\partial x_n^2} \); then

\[
\begin{align*}
0 &= \nabla^2 P \\
&= 2mnQ^{m-1}R + Q^m \nabla^2 R + 2 \sum_{i=1}^n 2mx_i Q^{m-1} \frac{\partial R}{\partial x_i} \\
&= Q^m \nabla^2 R + (2mn + 4md) Q^{m-1} R.
\end{align*}
\]

Since \( m > 0 \), this forces \( R \) to be divisible by \( Q \), contradiction.

Second solution: (by Noam Elkies) Retain notation as in the first solution. Let \( P_d \) be the set of homogeneous polynomials of degree \( d \), and let \( H_d \) be the subset of \( P_d \) of polynomials killed by \( \nabla^2 \), which has dimension \( \geq \dim(P_d) - \dim(P_{d-2}) \); the given problem amounts to showing that this inequality is actually an equality.

Consider the operator \( Q \nabla^2 \) (i.e., apply \( \nabla^2 \) then multiply by \( Q \)) on \( P_d \); its zero eigenspace is precisely \( H_d \). By the calculation from the first solution, if \( R \in P_d \), then

\[
\nabla^2(QR) - Q \nabla^2 R = (2n + 4d) R.
\]

Consequently, \( Q^j H_{d-2j} \) is contained in the eigenspace of \( Q \nabla^2 \) on \( P_d \) of eigenvalue

\[
(2n + 4(d - 2j)) + \cdots + (2n + 4(d - 2)).
\]

In particular, the \( Q^j H_{d-2j} \) lie in distinct eigenspaces, so are linearly independent within \( P_d \). But by dimension counting, their total dimension is at least that of \( P_d \). Hence they exhaust \( P_d \), and the zero eigenspace cannot have dimension greater than \( \dim(P_d) - \dim(P_{d-2}) \), as desired.

Third solution: (by Richard Stanley) Write \( x = (x_1, \ldots, x_n) \) and \( \nabla = (\frac{\partial}{\partial x_1}, \ldots, \frac{\partial}{\partial x_n}) \). Suppose that \( P(x) = Q(x)(x_1^2 + \cdots + x_n^2) \). Then

\[
P(\nabla)P(x) = Q(\nabla)(\nabla^2)P(x) = 0.
\]

On the other hand, if \( P(x) = \sum_\alpha c_\alpha x^\alpha \) (where \( \alpha = (\alpha_1, \ldots, \alpha_n) \) and \( x^\alpha = x_1^{\alpha_1} \cdots x_n^{\alpha_n} \)), then the constant term of \( P(\nabla)P(x) \) is seen to be \( \sum_\alpha c_\alpha^2 \). Hence \( c_\alpha = 0 \) for all \( \alpha \).

Remarks: The first two solutions apply directly over any field of characteristic zero. (The result fails in characteristic \( p > 0 \) because we may take \( P = (x_1^2 + \cdots + x_n^2)^p = x_1^{2p} + \cdots + x_n^{2p} \).) The third solution can be extended to complex coefficients by replacing \( P(\nabla) \) by its complex conjugate, and again the result may be deduced for any field of characteristic zero. Stanley also suggests Section 5 of the arXiv e-print math.CO/0502363 for some algebraic background for this problem.

B–6 First solution: Let \( I \) be the identity matrix, and let \( J_x \) be the matrix with \( x \)'s on the diagonal and 1's elsewhere. Note that \( J_x - (x-1)I \), being the all 1's matrix, has rank 1 and trace \( n \), so has \( n-1 \) eigenvalues equal to 0 and one equal to \( n \). Hence \( J_x \) has \( n-1 \) eigenvalues equal to \( x-1 \) and one equal to \( x+n-1 \), implying

\[
\det J_x = (x+n-1)(x-1)^{n-1}.
\]

On the other hand, we may expand the determinant as a sum indexed by permutations, in which case we get

\[
\det J_x = \sum_{\pi \in S_n} \operatorname{sgn}(\pi) x^{\nu(\pi)}.
\]

Integrating both sides from 0 to 1 (and substituting \( y = 1-x \)) yields

\[
\begin{align*}
\sum_{\pi \in S_n} \frac{\operatorname{sgn}(\pi)}{\nu(\pi)+1} &= \int_0^1 (x+n-1)(x-1)^{n-1} dx \\
&= \int_0^1 (-1)^{n+1}(1-y)(y-n)^{n-1} dy \\
&= (-1)^{n+1} \frac{n}{n+1},
\end{align*}
\]

as desired.

Second solution: We start by recalling a form of the principle of inclusion-exclusion: if \( f \) is a function on the power set of \( \{1, \ldots, n\} \), then

\[
f(S) = \sum_{T \supseteq S} (-1)^{|T|-|S|} \sum_{U \supseteq T} f(U).
\]

In this case we take \( f(S) \) to be the sum of \( \sigma(\pi) \) over all permutations \( \pi \) whose fixed points are exactly \( S \). Then \( \sum_{U \supseteq T} f(U) = 1 \) if \( |T| \geq n-1 \) and 0 otherwise (since a permutation group on 2 or more symbols has as many even and odd permutations), so

\[
f(S) = (-1)^{n-|S|}(1-n+|S|).
\]

The desired sum can thus be written, by grouping over fixed point sets, as

\[
\begin{align*}
\sum_{i=0}^n \binom{n}{i} (-1)^{n-i} \frac{1-n+i}{i+1} &= \sum_{i=0}^n (-1)^{n-i} \binom{n}{i} - \sum_{i=0}^n (-1)^{n-i} \frac{n}{i+1} \binom{n}{i} \\
&= 0 - \sum_{i=0}^n (-1)^{n-i} \frac{n}{n+1} \binom{n+1}{i+1} \\
&= (-1)^{n+1} \frac{n}{n+1}.
\end{align*}
\]