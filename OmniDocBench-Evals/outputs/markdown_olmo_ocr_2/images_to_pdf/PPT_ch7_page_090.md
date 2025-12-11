Cramer’s Rule for Linear Systems of Three Equations

(5)
\[
a_{11} x_1 + a_{12} x_2 + a_{13} x_3 = b_1 \\
a_{21} x_1 + a_{22} x_2 + a_{23} x_3 = b_2 \\
a_{31} x_1 + a_{32} x_2 + a_{33} x_3 = b_3
\]
is
(6)
\[
x_1 = \frac{D_1}{D}, \quad x_2 = \frac{D_2}{D}, \quad x_3 = \frac{D_3}{D} \qquad (D \neq 0)
\]
with the determinant \( D \) of the system given by (4) and
\[
D_1 = \begin{vmatrix}
b_1 & a_{12} & a_{13} \\
b_2 & a_{22} & a_{23} \\
b_3 & a_{32} & a_{33}
\end{vmatrix}, \quad
D_2 = \begin{vmatrix}
a_{11} & b_1 & a_{13} \\
a_{21} & b_2 & a_{23} \\
a_{31} & b_3 & a_{33}
\end{vmatrix}, \quad
D_3 = \begin{vmatrix}
a_{11} & a_{12} & b_1 \\
a_{21} & a_{22} & b_2 \\
a_{31} & a_{32} & b_3
\end{vmatrix}.
\]
Note that \( D_1, D_2, D_3 \) are obtained by replacing Columns 1, 2, 3, respectively, by the column of the right sides of (5).