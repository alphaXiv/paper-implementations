You must notice that the word “dimensional” is used in two different ways. We speak about a four-dimensional vector, meaning a vector in \( \mathbf{R}^4 \). Now we have defined a four-dimensional subspace; an example is the set of vectors in \( \mathbf{R}^6 \) whose first and last components are zero. The members of this four-dimensional subspace are six-dimensional vectors like (0, 5, 1, 3, 4, 0).

One final note about the language of linear algebra. We never use the terms “basis of a matrix” or “rank of a space” or “dimension of a basis.” These phrases have no meaning. It is the dimension of the column space that equals the rank of the matrix, as we prove in the coming section.

Problem Set 2.3

Problems 1–10 are about linear independence and linear dependence.

1. Show that \( v_1, v_2, v_3 \) are independent but \( v_1, v_2, v_3, v_4 \) are dependent:

\[
v_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}, \quad v_4 = \begin{bmatrix} 2 \\ 3 \\ 4 \end{bmatrix}.
\]

Solve \( c_1 v_1 + \cdots + c_4 v_4 = 0 \) or \( Ac = 0 \). The \( v \)'s go in the columns of \( A \).

2. Find the largest possible number of independent vectors among

\[
v_1 = \begin{bmatrix} 1 \\ -1 \\ 0 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 1 \\ 0 \\ -1 \\ 0 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ -1 \end{bmatrix}, \quad v_4 = \begin{bmatrix} 0 \\ 1 \\ -1 \\ 0 \end{bmatrix}, \quad v_5 = \begin{bmatrix} 0 \\ 1 \\ 0 \\ -1 \end{bmatrix}, \quad v_6 = \begin{bmatrix} 0 \\ 0 \\ 1 \\ -1 \end{bmatrix}.
\]

This number is the ____ of the space spanned by the \( v \)'s.

3. Prove that if \( a = 0, d = 0, \) or \( f = 0 \) (3 cases), the columns of \( U \) are dependent:

\[
U = \begin{bmatrix} a & b & c \\ 0 & d & e \\ 0 & 0 & f \end{bmatrix}.
\]

4. If \( a, d, f \) in Problem 3 are all nonzero, show that the only solution to \( Ux = 0 \) is \( x = 0 \). Then \( U \) has independent columns.

5. Decide the dependence or independence of

(a) the vectors (1, 3, 2), (2, 1, 3), and (3, 2, 1).

(b) the vectors (1, −3, 2), (2, 1, −3), and (−3, 2, 1).