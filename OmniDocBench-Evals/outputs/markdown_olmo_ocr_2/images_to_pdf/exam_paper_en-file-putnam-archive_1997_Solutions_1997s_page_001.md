Solutions to the 58th William Lowell Putnam Mathematical Competition
Saturday, December 6, 1997

Manjul Bhargava, Kiran Kedlaya, and Lenny Ng

A–1 The centroid G of the triangle is collinear with H and O (Euler line), and the centroid lies two-thirds of the way from A to M. Therefore H is also two-thirds of the way from A to F, so \( AF = 15 \). Since the triangles BFH and AFC are similar (they’re right triangles and

\[
\angle HBC = \pi/2 - \angle C = \angle CAF),
\]

we have

\[
BF/FH = AF/FC
\]

or

\[
BF \cdot FC = FH \cdot AF = 75.
\]

Now

\[
BC^2 = (BF + FC)^2 = (BF - FC)^2 + 4BF \cdot FC,
\]

but

\[
BF - FC = BM + MF - (MC - MF) = 2MF = 22,
\]

so

\[
BC = \sqrt{22^2 + 4 \cdot 75} = \sqrt{784} = 28.
\]

A–2 We show more precisely that the game terminates with one player holding all of the pennies if and only if \( n = 2^m + 1 \) or \( n = 2^m + 2 \) for some \( m \). First suppose we are in the following situation for some \( k \geq 2 \). (Note: for us, a “move” consists of two turns, starting with a one-penny pass.)

– Except for the player to move, each player has \( k \) pennies;
– The player to move has at least \( k \) pennies.

We claim then that the game terminates if and only if the number of players is a power of 2. First suppose the number of players is even; then after \( m \) complete rounds, every other player, starting with the player who moved first, will have \( m \) more pennies than initially, and the others will all have 0. Thus we are reduced to the situation with half as many players; by this process, we eventually reduce to the case where the number of players is odd. However, if there is more than one player, after two complete rounds everyone has as many pennies as they did before (here we need \( m \geq 2 \)), so the game fails to terminate. This verifies the claim.

Returning to the original game, note that after one complete round, \( \left\lfloor \frac{n-1}{2} \right\rfloor \) players remain, each with 2 pennies except for the player to move, who has either 3 or 4 pennies. Thus by the above argument, the game terminates if and only if \( \left\lfloor \frac{n-1}{2} \right\rfloor \) is a power of 2, that is, if and only if \( n = 2^m + 1 \) or \( n = 2^m + 2 \) for some \( m \).

A–3 Note that the series on the left is simply \( x \exp(-x^2/2) \). By integration by parts,

\[
\int_0^\infty x^{2n+1} e^{-x^2/2} dx = 2n \int_0^\infty x^{2n-1} e^{-x^2/2} dx
\]

and so by induction,

\[
\int_0^\infty x^{2n+1} e^{-x^2/2} dx = 2 \times 4 \times \cdots \times 2n.
\]

Thus the desired integral is simply

\[
\sum_{n=0}^\infty \frac{1}{2^n n!} = \sqrt{e}.
\]

A–4 In order to have \( \psi(x) = a \phi(x) \) for all \( x \), we must in particular have this for \( x = e \), and so we take \( a = \phi(e)^{-1} \). We first note that

\[
\phi(g) \phi(e) \phi(g^{-1}) = \phi(e) \phi(g) \phi(g^{-1})
\]

and so \( \phi(g) \) commutes with \( \phi(e) \) for all \( g \). Next, we note that

\[
\phi(x) \phi(y) \phi(y^{-1} x^{-1}) = \phi(e) \phi(xy) \phi(y^{-1} x^{-1})
\]

and using the commutativity of \( \phi(e) \), we deduce

\[
\phi(e)^{-1} \phi(x) \phi(e)^{-1} \phi(y) = \phi(e)^{-1} \phi(xy)
\]

or \( \psi(xy) = \psi(x) \psi(y) \), as desired.

A–5 We may discard any solutions for which \( a_1 \neq a_2 \), since those come in pairs; so assume \( a_1 = a_2 \). Similarly, we may assume that \( a_3 = a_4, a_5 = a_6, a_7 = a_8, a_9 = a_{10} \). Thus we get the equation

\[
2/a_1 + 2/a_3 + 2/a_5 + 2/a_7 + 2/a_9 = 1.
\]

Again, we may assume \( a_1 = a_3 \) and \( a_5 = a_7 \), so we get \( 4/a_1 + 4/a_5 + 2/a_9 = 1 \); and \( a_1 = a_5 \), so \( 8/a_1 + 2/a_9 = 1 \). This implies that \( (a_1 - 8)(a_9 - 2) = 16 \), which by counting has 5 solutions. Thus \( N_{10} \) is odd.

A–6 Clearly \( x_{n+1} \) is a polynomial in \( c \) of degree \( n \), so it suffices to identify \( n \) values of \( c \) for which \( x_{n+1} = 0 \). We claim these are \( c = n - 1 - 2r \) for \( r = 0, 1, \ldots, n-1 \); in this case, \( x_r \) is the coefficient of \( t^{k-1} \) in the polynomial \( f(t) = (1-t)^r (1+t)^{n-1-r} \). This can be verified by noticing that \( f \) satisfies the differential equation

\[
\frac{f'(t)}{f(t)} = \frac{n-1-r}{1+t} - \frac{r}{1-t}
\]