identity matrix), the bottom right block is \( M_{n-2k-1,k} \), and the other two blocks are zero. We conclude that

\[
\det(M_{n,k}) \equiv \det(M_{n-2k-1,k}) \pmod{2},
\]

proving the desired congruence.

To prove the desired result, we must now check that \( F_{0,k}, F_{1,k} \) are odd and \( F_{2,k}, \ldots, F_{2k,k} \) are even. For \( n = 0, \ldots, k+1 \), the matrix \( M_{n,k} \) consists of all ones, so its determinant is 1 if \( n = 0,1 \) and 0 otherwise. (Alternatively, we have \( F_{n,k} = n! \) for \( n = 0, \ldots, k+1 \), since every permutation of \( \{1, \ldots, n\} \) is \( k \)-limited.) For \( n = k+2, \ldots, 2k \), observe that rows \( k \) and \( k+1 \) of \( M_{n,k} \) both consist of all ones, so \( \det(M_{n,k}) = 0 \) as desired.

**Third solution:** (by Tom Belulovich) Define \( M_{n,k} \) as in the second solution. We prove \( \det(M_{n,k}) \) is odd for \( n \equiv 0,1 \pmod{2k+1} \) and even otherwise, by directly determining whether or not \( M_{n,k} \) is invertible as a matrix over the field of two elements.

Let \( r_i \) denote row \( i \) of \( M_{n,k} \). We first check that if \( n \equiv 2, \ldots, 2k \pmod{2k+1} \), then \( M_{n,k} \) is not invertible. In this case, we can find integers \( 0 \leq a < b \leq k \) such that \( n + a + b \equiv 0 \pmod{2k+1} \). Put \( j = (n + a + b)/(2k+1) \). We can then write the all-ones vector both as

\[
\sum_{i=0}^{j-1} r_{k+1-a+(2k+1)i}
\]

and as

\[
\sum_{i=0}^{j-1} r_{k+1-b+(2k+1)i}.
\]

Hence \( M_{n,k} \) is not invertible.

We next check that if \( n \equiv 0,1 \pmod{2k+1} \), then \( M_{n,k} \) is invertible. Suppose that \( a_1, \ldots, a_n \) are scalars such that \( a_1 r_1 + \cdots + a_n r_n \) is the zero vector. The \( m \)-th coordinate of this vector equals \( a_{m-k} + \cdots + a_{m+k} \), where we regard \( a_i \) as zero if \( i \notin \{1, \ldots, n\} \). By comparing consecutive coordinates, we obtain

\[
a_{m-k} = a_{m+k+1} \qquad (1 \leq m < n).
\]

In particular, the \( a_i \) repeat with period \( 2k+1 \). Taking \( m = 1, \ldots, k \) further yields that

\[
a_{k+2} = \cdots = a_{2k+1} = 0
\]

while taking \( m = n-k, \ldots, n-1 \) yields

\[
a_{n-2k} = \cdots = a_{n-1-k} = 0.
\]

For \( n \equiv 0 \pmod{2k+1} \), the latter can be rewritten as

\[
a_1 = \cdots = a_k = 0
\]

whereas for \( n \equiv 1 \pmod{2k+1} \), it can be rewritten as

\[
a_2 = \cdots = a_{k+1} = 0.
\]

In either case, since we also have

\[
a_1 + \cdots + a_{2k+1} = 0
\]

from the \( (k+1) \)-st coordinate, we deduce that all of the \( a_i \) must be zero, and so \( M_{n,k} \) must be invertible.

**Remark:** The matrices \( M_{n,k} \) are examples of *banded matrices*, which occur frequently in numerical applications of linear algebra. They are also examples of *Toeplitz matrices*.