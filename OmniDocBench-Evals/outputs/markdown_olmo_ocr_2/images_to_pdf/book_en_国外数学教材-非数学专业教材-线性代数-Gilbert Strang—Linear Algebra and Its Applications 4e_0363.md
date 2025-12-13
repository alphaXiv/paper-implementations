Thus \( A_k \) is positive definite. Its eigenvalues (not the same \( \lambda_1! \)) must be positive. Its determinant is their product, so all upper left determinants are positive.

*If condition III holds, so does condition IV:* According to Section 4.4, the kth pivot \( d_k \) is the ratio of \( \det A_k \) to \( \det A_{k-1} \). If the determinants are all positive, so are the pivots.

*If condition IV holds, so does condition I:* We are given positive pivots, and must deduce that \( x^T A x > 0 \). This is what we did in the 2 by 2 case, by completing the square. The pivots were the numbers outside the squares. To see how that happens for symmetric matrices of any size, we go back to *elimination on a symmetric matrix*: \( A = LDL^T \).

**Example 1.** Positive pivots 2, \( \frac{3}{2} \), and \( \frac{4}{3} \):

\[
A = \begin{bmatrix}
2 & -1 & 0 \\
-1 & 2 & -1 \\
0 & -1 & 2
\end{bmatrix} =
\begin{bmatrix}
1 & 0 & 0 \\
-\frac{1}{2} & 1 & 0 \\
0 & -\frac{2}{3} & 1
\end{bmatrix}
\begin{bmatrix}
2 \\
\frac{3}{2} \\
\frac{4}{3}
\end{bmatrix}
\begin{bmatrix}
1 & -\frac{1}{2} & 0 \\
0 & 1 & -\frac{2}{3} \\
0 & 0 & 1
\end{bmatrix} = LDL^T.
\]

I want to split \( x^T A x \) into \( x^T LDL^T x \):

\[
\text{If } x = \begin{bmatrix} u \\ v \\ w \end{bmatrix}, \quad \text{then} \quad L^T x = \begin{bmatrix} 1 & -\frac{1}{2} & 0 \\ 0 & 1 & -\frac{2}{3} \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} u \\ v \\ w \end{bmatrix} = \begin{bmatrix} u - \frac{1}{2} v \\ v - \frac{2}{3} w \\ w \end{bmatrix}.
\]

So \( x^T A x \) is a sum of squares with the pivots 2, \( \frac{3}{2} \), and \( \frac{4}{3} \) as coefficients:

\[
x^T A x = (L^T x)^T D (L^T x) = 2 \left( u - \frac{1}{2} v \right)^2 + \frac{3}{2} \left( v - \frac{2}{3} w \right)^2 + \frac{4}{3} (w)^2.
\]

Those positive pivots in \( D \) multiply perfect squares to make \( x^T A x \) positive. Thus condition IV implies condition I, and the proof is complete.

It is beautiful that elimination and completing the square are actually the same. Elimination removes \( x_1 \) from all later equations. Similarly, the first square accounts for all terms in \( x^T A x \) involving \( x_1 \). The sum of squares has the pivots outside. *The multipliers \( \ell_{ij} \) are inside!* You can see the numbers \( -\frac{1}{2} \) and \( -\frac{2}{3} \) inside the squares in the example.

*Every diagonal entry \( a_{ii} \) must be positive.* As we know from the examples, however, it is far from sufficient to look only at the diagonal entries.

The pivots \( d_i \) are not to be confused with the eigenvalues. For a typical positive definite matrix, they are two completely different sets of positive numbers, In our 3 by 3 example, probably the determinant test is the easiest:

**Determinant test** \( \det A_1 = 2,\quad \det A_2 = 3,\quad \det A_3 = \det A = 4. \)

The pivots are the ratios \( d_1 = 2,\ d_2 = \frac{3}{2},\ d_3 = \frac{4}{3} \). Ordinarily the eigenvalue test is the longest computation. For this \( A \) we know the \( \lambda \)'s are all positive:

**Eigenvalue test** \( \lambda_1 = 2 - \sqrt{2},\quad \lambda_2 = 2,\quad \lambda_3 = 2 + \sqrt{2}. \)