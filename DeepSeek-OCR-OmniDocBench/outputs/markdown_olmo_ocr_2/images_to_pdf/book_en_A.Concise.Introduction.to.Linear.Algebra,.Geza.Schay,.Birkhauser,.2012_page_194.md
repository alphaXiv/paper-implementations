\[
S(\mathbf{x}) = \begin{bmatrix} x_2 + x_3 \\ x_3 \end{bmatrix}
\] (4.62)

and

\[
T(\mathbf{x}) = \begin{bmatrix} x_1 - x_2 \\ x_1 + x_3 \end{bmatrix}.
\] (4.63)

Then \( Q = S + T \) and \( R = cT \) are the linear transformations given by

\[
Q(\mathbf{x}) = \begin{bmatrix} x_1 + x_3 \\ x_1 + 2x_3 \end{bmatrix}
\] (4.64)

and

\[
R(\mathbf{x}) = \begin{bmatrix} c(x_1 - x_2) \\ c(x_1 + x_3) \end{bmatrix}.
\] (4.65)

With Definition 4.2.4 Theorems 4.2.8 and 4.2.9 lead to the following theorem.

**Theorem 4.2.10. (The Linear Transformations from U to V Form a Vector Space, Which Is Isomorphic to \( \mathcal{M}_{m,n} \)).** Let \( L(U,V) \) denote the set of all linear transformations from an \( n \)-dimensional vector space \( U \) to an \( m \)-dimensional vector space \( V \), and let \( A = (\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n) \) be an ordered basis for \( U \), \( B = (\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_m) \) an ordered basis for \( V \), and \( T_{A,B} \) the \( m \times n \) matrix that represents any \( T \in L(U,V) \) relative to these bases. Then
1. \( L(U,V) \), together with addition and scalar multiple of transformations as in Definition 4.2.4, is a vector space.
2. The mapping \( M \) from \( L(U,V) \) to the vector space \( \mathcal{M}_{m,n} \) of all \( m \times n \) matrices\footnote{See Example 3.1.2.} given by \( M(T) = T_{A,B} \) is linear and an isomorphism. Hence \( L(U,V) \) is \( mn \)-dimensional.

*Proof.* 1. \( L(U,V) \) is clearly nonempty: the zero mapping \( O \) is in it. Theorem 4.2.9 shows that \( L(U,V) \) is closed under addition and multiplication by scalars. The vector space axioms for \( L(U,V) \) follow from the corresponding ones in \( V \) for every \( \mathbf{x} \) in \( \mathbf{y} = T(\mathbf{x}) \). In particular, the zero element is the zero mapping \( O \), and the element \( -T \) is the mapping \( (-1)T \).

2. Let \( S, T \in L(U,V) \) and \( a, b \) any scalars. Then, by Theorem 4.1.3, for all \( \mathbf{x} \in U \), \( \mathbf{y} = T(\mathbf{x}) \) becomes in terms of coordinates \( \mathbf{y}_B = T_{A,B} \mathbf{x}_A = M(T) \mathbf{x}_A \). Similarly \( \mathbf{y} = (aS + bT)(\mathbf{x}) \) becomes

\[
\mathbf{y}_B = (aS + bT)_{A,B} \mathbf{x}_A = M(aS + bT) \mathbf{x}_A.
\] (4.66)