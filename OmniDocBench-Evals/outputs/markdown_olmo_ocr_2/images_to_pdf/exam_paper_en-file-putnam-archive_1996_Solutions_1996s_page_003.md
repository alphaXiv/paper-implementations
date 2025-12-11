tiating and taking square roots,

\[
\left( \frac{2n-1}{e} \right)^{\frac{2n-1}{2}} < (2n-1)^{\frac{2n-1}{2}} e^{-n+1}
\]
\[
\leq 1 \cdot 3 \cdots (2n-1)
\]
\[
\leq (2n+1)^{\frac{2n+1}{2}} \frac{e^{-n+1}}{3^{3/2}}
\]
\[
< \left( \frac{2n+1}{e} \right)^{\frac{2n+1}{2}},
\]

using the fact that \(1 < e < 3\).

B–3 View \(x_1, \ldots, x_n\) as an arrangement of the numbers \(1, 2, \ldots, n\) on a circle. We prove that the optimal arrangement is

\[
\ldots, n-4, n-2, n, n-1, n-3, \ldots
\]

To show this, note that if \(a, b\) is a pair of adjacent numbers and \(c, d\) is another pair (read in the same order around the circle) with \(a < d\) and \(b > c\), then the segment from \(b\) to \(c\) can be reversed, increasing the sum by

\[
ac + bd - ab - cd = (d-a)(b-c) > 0.
\]

Now relabel the numbers so they appear in order as follows:

\[
\ldots, a_{n-4}, a_{n-2}, a_n = n, a_{n-1}, a_{n-3}, \ldots
\]

where without loss of generality we assume \(a_{n-1} > a_{n-2}\). By considering the pairs \(a_{n-2}, a_n\) and \(a_{n-1}, a_{n-3}\) and using the trivial fact \(a_i > a_{i-1}\), we deduce \(a_{n-2} > a_{n-3}\). We then compare the pairs \(a_{n-4}, a_{n-2}\) and \(a_{n-1}, a_{n-3}\), and using that \(a_{n-1} > a_{n-2}\), we deduce \(a_{n-3} > a_{n-4}\). Continuing in this fashion, we prove that \(a_n > a_{n-1} > \cdots > a_1\) and so \(a_k = k\) for \(k = 1, 2, \ldots, n\), i.e. that the optimal arrangement is as claimed. In particular, the maximum value of the sum is

\[
1 \cdot 2 + (n-1) \cdot n + 1 \cdot 3 + 2 \cdot 4 + \cdots + (n-2) \cdot n
\]
\[
= 2 + n^2 - n + (1^2 - 1) + \cdots + [(n-1)^2 - 1]
\]
\[
= n^2 - n + 2 - (n-1) + \frac{(n-1)n(2n-1)}{6}
\]
\[
= \frac{2n^3 + 3n^2 - 11n + 18}{6}.
\]

Alternate solution: We prove by induction that the value given above is an upper bound; it is clearly a lower bound because of the arrangement given above. Assume this is the case for \(n-1\). The optimal arrangement for \(n\) is obtained from some arrangement for \(n-1\) by inserting \(n\) between some pair \(x, y\) of adjacent terms. This operation increases the sum by \(nx + ny - xy = n^2 - (n-x)(n-y)\), which is an increasing function of both \(x\) and \(y\). In particular, this difference is maximal when \(x\) and \(y\) equal \(n-1\) and \(n-2\). Fortunately, this yields precisely the difference between the claimed upper bound for \(n\) and the assumed upper bound for \(n-1\), completing the induction.

B–4 Suppose such a matrix \(A\) exists. If the eigenvalues of \(A\) (over the complex numbers) are distinct, then there exists a complex matrix \(C\) such that \(B = CAC^{-1}\) is diagonal. Consequently, \(\sin B\) is diagonal. But then \(\sin A = C^{-1} (\sin B) C\) must be diagonalizable, a contradiction. Hence the eigenvalues of \(A\) are the same, and \(A\) has a conjugate \(B = CAC^{-1}\) over the complex numbers of the form

\[
\begin{pmatrix}
x & y \\
0 & x
\end{pmatrix}.
\]

A direct computation shows that

\[
\sin B = \begin{pmatrix}
\sin x & y \cdot \cos x \\
0 & \sin x
\end{pmatrix}.
\]

Since \(\sin A\) and \(\sin B\) are conjugate, their eigenvalues must be the same, and so we must have \(\sin x = 1\). This implies \(\cos x = 0\), so that \(\sin B\) is the identity matrix, as must be \(\sin A\), a contradiction. Thus \(A\) cannot exist.

Alternate solution (due to Craig Helfgott and Alex Popa): Define both \(\sin A\) and \(\cos A\) by the usual power series. Since \(A\) commutes with itself, the power series identity

\[
\sin^2 A + \cos^2 A = I
\]

holds. But if \(\sin A\) is the given matrix, then by the above identity, \(\cos^2 A\) must equal \(
\begin{pmatrix}
0 & -2 \cdot 1996 \\
0 & 0
\end{pmatrix}
\) which is a nilpotent matrix. Thus \(\cos A\) is also nilpotent. However, the square of any \(2 \times 2\) nilpotent matrix must be zero (e.g., by the Cayley-Hamilton theorem). This is a contradiction.

B–5 Consider a \(1 \times n\) checkerboard, in which we write an \(n\)-letter string, one letter per square. If the string is balanced, we can cover each pair of adjacent squares containing the same letter with a \(1 \times 2\) domino, and these will not overlap (because no three in a row can be the same). Moreover, any domino is separated from the next by an even number of squares, since they must cover opposite letters, and the sequence must alternate in between.

Conversely, any arrangement of dominoes where adjacent dominoes are separated by an even number of squares corresponds to a unique balanced string, once we choose whether the string starts with \(X\) or \(O\). In other words, the number of balanced strings is twice the number of acceptable domino arrangements.

We count these arrangements by numbering the squares \(0, 1, \ldots, n-1\) and distinguishing whether the dominoes start on even or odd numbers. Once this is decided, one