(e) the “reverse-triangular” matrix that results from row exchanges,

\[
M = \begin{bmatrix}
0 & 0 & 0 & 2 \\
0 & 0 & 2 & 6 \\
0 & 1 & 2 & 2 \\
4 & 4 & 8 & 8
\end{bmatrix}.
\]

8. Show how rule 6 (\(\det = 0\) if a row is zero) comes directly from rules 2 and 3.

9. Suppose you do two row operations \(at once\), going from

\[
\begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \text{to} \quad \begin{bmatrix} a - mc & b - md \\ c - \ell a & d - \ell b \end{bmatrix}.
\]

Find the determinant of the new matrix, by rule 3 or by direct calculation.

10. If \(Q\) is an orthogonal matrix, so that \(Q^T Q = I\), prove that \(\det Q\) equals \(+1\) or \(-1\). What kind of box is formed from the rows (or columns) of \(Q\)?

11. Prove again that \(\det Q = 1\) or \(-1\) using only the Product rule. If \(|\det Q| > 1\) then \(\det Q^n\) blows up. How do you know this can’t happen to \(Q^n\)?

12. Use row operations to verify that the 3 by 3 “Vandermonde determinant” is

\[
\det \begin{bmatrix} 1 & a & a^2 \\ 1 & b & b^2 \\ 1 & c & c^2 \end{bmatrix} = (b-a)(c-a)(c-b).
\]

13. (a) A skew-symmetric matrix satisfies \(K^T = -K\), as in

\[
K = \begin{bmatrix} 0 & a & b \\ -a & 0 & c \\ -b & -c & 0 \end{bmatrix}.
\]

In the 3 by 3 case, why is \(\det(-K) = (-1)^3 \det K\)? On the other hand \(\det K^T = \det K\) (always). Deduce that the determinant must be zero.

(b) Write down a 4 by 4 skew-symmetric matrix with \(\det K\) not zero.

14. True or false, with reason if true and counterexample if false:

(a) If \(A\) and \(B\) are identical except that \(b_{11} = 2a_{11}\), then \(\det B = 2\det A\).

(b) The determinant is the product of the pivots.

(c) If \(A\) is invertible and \(B\) is singular, then \(A + B\) is invertible.

(d) If \(A\) is invertible and \(B\) is singular, then \(AB\) is singular.

(e) The determinant of \(AB - BA\) is zero.