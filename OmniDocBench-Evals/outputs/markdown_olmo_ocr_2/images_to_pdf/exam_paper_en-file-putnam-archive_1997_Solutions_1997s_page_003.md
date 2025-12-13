we have

\[
s_k - s_{k-1} + s_{k+2} = \sum_i (-1)^i (a_{n-i,i} + a_{n-i,i+1} + a_{n-i,i+2})
= \sum_i (-1)^i a_{n-i+1,i+2} = s_{k+3}.
\]

By computing \( s_0 = 1, s_1 = 1, s_2 = 0 \), we may easily verify by induction that \( s_{4j} = s_{4j+1} = 1 \) and \( s_{4j+2} = s_{4j+3} = 0 \) for all \( j \geq 0 \). (Alternate solution suggested by John Rickert: write \( S(x,y) = \sum_{j=0}^{\infty} (y + xy^2 + x^2 y^3)^j \), and note note that \( s_k \) is the coefficient of \( y^k \) in \( S(-1,y) = (1+y)/(1-y^4) \).)

B–5 Define the sequence \( x_1 = 2, x_n = 2^{x_{n-1}} \) for \( n > 1 \). It suffices to show that for every \( n \), \( x_m \equiv x_{m+1} \equiv \cdots \pmod{n} \) for some \( m < n \). We do this by induction on \( n \), with \( n = 2 \) being obvious.

Write \( n = 2^a b \), where \( b \) is odd. It suffices to show that \( x_m \equiv \cdots \pmod{2^a} \) and modulo \( b \), for some \( m < n \). For the former, we only need \( x_{n-1} \geq a \), but clearly \( x_{n-1} \geq n \) by induction on \( n \). For the latter, note that \( x_m \equiv x_{m+1} \equiv \cdots \pmod{b} \) as long as \( x_{m-1} \equiv x_m \equiv \cdots \pmod{\phi(b)} \), where \( \phi(n) \) is the Euler totient function. By hypothesis, this occurs for some \( m < \phi(b) + 1 \leq n \). (Thanks to Anoop Kulkarni for catching a lethal typo in an earlier version.)

B–6 The answer is 25/13. Place the triangle on the cartesian plane so that its vertices are at \( C = (0,0), A = (0,3), B = (4,0) \). Define also the points \( D = (20/13, 24/13) \), and \( E = (27/13, 0) \). We then compute that

\[
\frac{25}{13} = AD = BE = DE \\
\frac{27}{13} = BC - CE = BE < BC \\
\frac{39}{13} = AC < \sqrt{AC^2 + CE^2} = AE \\
\frac{40}{13} = AB - AD = BD < AB
\]

and that \( AD < CD \). In any dissection of the triangle into four parts, some two of \( A, B, C, D, E \) must belong to the same part, forcing the least diameter to be at least 25/13.

We now exhibit a dissection with least diameter 25/13. (Some variations of this dissection are possible.) Put \( F = (15/13, 19/13) \), \( G = (15/13, 0) \), \( H = (0, 19/13) \), \( J = (32/15, 15/13) \), and divide \( ABC \) into the convex polygonal regions \( ADFH, BEJ, CGFH, DFGEJ \). To check that this dissection has least diameter 25/13, it suffices (by the following remark) to check that the distances

\[
AD, AF, AH, BE, BJ, DE, CF, CG, CH, DF, DG, DH, DJ, EF, EG, EJ, FG, FH, FJ, GJ
\]

are all at most 25/13. This can be checked by a long numerical calculation, which we omit in favor of some shortcuts: note that \( ADFH \) and \( BEJ \) are contained in circular sectors centered at \( A \) and \( B \), respectively, of radius 25/13 and angle less than \( \pi/3 \), while \( CGFH \) is a rectangle with diameter \( CF < 25/13 \).

Remark. The preceding argument uses implicitly the fact that for \( P \) a simple closed polygon in the plane, if we let \( S \) denote the set of points on or within \( P \), then the maximum distance between two points of \( S \) occurs between some pair of vertices of \( P \). This is an immediate consequence of the compactness of \( S \) (which guarantees the existence of a maximum) and the convexity of the function taking \( (x,y) \in S \times S \) to the squared distance between \( x \) and \( y \) (which is obvious in terms of Cartesian coordinates).