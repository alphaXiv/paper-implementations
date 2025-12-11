• \( C \) and \( D \) are at the same side of \( AB \): if \( ACD \) was to be acute, then \( AC \) or \( AD \) would leave \( B \) and \( O \) at different half-planes. In turn, this would assure that the distance from \( AC \) or \( AD \) to \( O \) would be less than the distance of \( AB \) to \( O \), thus contradicting our choice of \( A \) and \( B \). Therefore, \( ACD \) is not acute and, accordingly, the same is true of \( BCD \).
• \( C \) and \( D \) are in opposite sides with respect to \( AB \): since \( ACBD \) is convex, at most one of the triangles \( ACD \) or \( BCD \) contains the center \( O \), and hence at most one of such triangles is acute. On the other hand, by the choice of \( A \) and \( B \) the set \( \{C, D\} \) can be chosen in exactly \( k(k - 1) \) ways, and this is the largest possible number of acute triangles of this kind.

(iii) Those having neither \( A \) nor \( B \) as one of their vertices: by induction hypothesis, there are at most \( \frac{1}{6}(k - 1)k(2k - 1) \) such triangles.

Adding the contributions of the three cases above, we conclude that the number of acute triangles having its vertices at three of the \( 2k + 1 \) given points is at most

\[
k + k(k - 1) + \frac{1}{6}(k - 1)k(2k - 1) = \frac{1}{6}k(k + 1)(2k + 1).
\]

Problems: Sect. 4.2

1. (India) Prove that, for every \( n \geq 6 \), every square can be partitioned into \( n \) other squares.
2. (Brazil) Given a natural number \( n > 1 \), we write a real number of modulus less than 1 in each cell of an \( n \times n \) chessboard, in such a way that the sum of the numbers written in the four cells of any \( 2 \times 2 \) square is equal to 0. If \( n \) is odd, show that the sum of the numbers written in the \( n^2 \) cells is less than \( n \).
3. (TT) Point \( O \) is situated in the interior of the convex polygon \( A_1A_2 \ldots A_n \). Consider all of the angles \( \angle A_i OA_j \), with distinct \( 1 \leq i, j \leq n \). Prove that at least \( n - 1 \) of them are not acute.
4. (TT) In a convex polygon \( P \) some diagonals were drawn, such that no two of them intersect in the interior of \( P \). Show that there are at least two vertices of \( P \) such that none of the traced diagonals is incident to none of these two vertices.
5. (Hungary) In an \( n \times n \) chessboard we have a certain number of towers (at most one tower for each \( 1 \times 1 \) square). A tower can move from one square to another if and only if these two squares belong to a single line or column of the board and there are no other towers in between. On the other hand, if two towers belong to a single line or column and there are no other towers in between, then we say that any of these towers can attack the other. Prove that one can paint the towers with one of three colors in such a way that no tower can attack another tower of the same color.