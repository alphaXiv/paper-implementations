by Stirling’s approximation is asymptotic to a constant times \( m^{-5/2} \). This term alone is bigger than \( c(2/3)^n \), so we must have \( A_{n+1}/A_n \geq 2/3 \) for some \( n \). (In fact, we must have \( A_{n+1}/A_n \geq 1 - \varepsilon \) for any \( \varepsilon > 0 \).)

B–1 For a given \( \pi \), no more than three different values of \( \pi(x) \) are possible (four would require one part each of size at least 1,2,3,4, and that’s already more than 9 elements). If no such \( x, y \) exist, each pair \( (\pi(x), \pi'(x)) \) occurs for at most 1 element of \( x \), and since there are only \( 3 \times 3 \) possible pairs, each must occur exactly once. In particular, each value of \( \pi(x) \) must occur 3 times. However, clearly any given value of \( \pi(x) \) occurs \( k\pi(x) \) times, where \( k \) is the number of distinct partitions of that size. Thus \( \pi(x) \) can occur 3 times only if it equals 1 or 3, but we have three distinct values for which it occurs, contradiction.

B–2 For those who haven’t taken enough physics, “rolling without slipping” means that the perimeter of the ellipse and the curve pass at the same rate, so all we’re saying is that the perimeter of the ellipse equals the length of one period of the sine curve. So set up the integrals:

\[
\int_0^{2\pi} \sqrt{(-a \sin \theta)^2 + (b \cos \theta)^2} d\theta
= \int_0^{2\pi a} \sqrt{1 + (c/a \cos x/a)^2} dx.
\]

Let \( \theta = x/a \) in the second integral and write 1 as \( \sin^2 \theta + \cos^2 \theta \) and you get

\[
\int_0^{2\pi} \sqrt{a^2 \sin^2 \theta + b^2 \cos^2 \theta} d\theta
= \int_0^{2\pi} \sqrt{a^2 \sin^2 \theta + (a^2 + c^2) \cos^2 \theta} d\theta.
\]

Since the left side is increasing as a function of \( b \), we have equality if and only if \( b^2 = a^2 + c^2 \).

B–3 For \( n = 1 \) we obviously get 45, while for \( n = 3 \) the answer is 0 because it both changes sign (because determinants are alternating) and remains unchanged (by symmetry) when you switch any two rows other than the first one. So only \( n = 2 \) is left. By the multilinearity of the determinant, the answer is the determinant of the matrix whose first (resp. second) row is the sum of all possible first (resp. second) rows. There are 90 first rows whose sum is the vector (450, 405), and 100 second rows whose sum is (450, 450). Thus the answer is \( 450 \times 450 - 450 \times 405 = 45 \times 450 = 20250 \).

B–4 The infinite continued fraction is defined as the limit of the sequence \( L_0 = 2207, L_{n+1} = 2207 - 1/L_n \). Notice that the sequence is strictly decreasing (by induction) and thus indeed has a limit \( L \), which satisfies \( L = 2207 - 1/L \), or rewriting, \( L^2 - 2207L + 1 = 0 \). Moreover, we want the greater of the two roots.

Now how to compute the eighth root of \( L \)? Notice that if \( x \) satisfies the quadratic \( x^2 - ax + 1 = 0 \), then we have

\[
0 = (x^2 - ax + 1)(x^2 + ax + 1)
= x^4 - (a^2 - 2)x^2 + 1.
\]

Clearly, then, the positive square roots of the quadratic \( x^2 - bx + 1 \) satisfy the quadratic \( x^2 - (b^2 + 2)^{1/2}x + 1 = 0 \). Thus we compute that \( L^{1/2} \) is the greater root of \( x^2 - 47x + 1 = 0 \), \( L^{1/4} \) is the greater root of \( x^2 - 7x + 1 = 0 \), and \( L^{1/8} \) is the greater root of \( x^2 - 3x + 1 = 0 \), otherwise known as \( (3 + \sqrt{5})/2 \).

B–5 This problem is dumb if you know the Sprague-Grundy theory of normal impartial games (see Conway, Berlekamp and Guy, Winning Ways, for details). I’ll describe how it applies here. To each position you assign a nim-value as follows. A position with no moves (in which case the person to move has just lost) takes value 0. Any other position is assigned the smallest number not assigned to a valid move from that position.

For a single pile, one sees that an empty pile has value 0, a pile of 2 has value 1, a pile of 3 has value 2, a pile of 4 has value 0, a pile of 5 has value 1, and a pile of 6 has value 0.

You add piles just like in standard Nim: the nim-value of the composite of two games (where at every turn you pick a game and make a move there) is the “base 2 addition without carries” (i.e. exclusive OR) of the nim-values of the constituents. So our starting position, with piles of 3, 4, 5, 6, has nim-value \( 2 \oplus 0 \oplus 1 \oplus 0 = 3 \).

A position is a win for the player to move if and only if it has a nonzero value, in which case the winning strategy is to always move to a 0 position. (This is always possible from a nonzero position and never from a zero position, which is precisely the condition that defines the set of winning positions.) In this case, the winning move is to reduce the pile of 3 down to 2, and you can easily describe the entire strategy if you so desire.

B–6 Obviously \( \alpha, \beta, \gamma \) have to be greater than 1, and no two can both be rational, so without loss of generality assume that \( \alpha \) and \( \beta \) are irrational. Let \( \{x\} = x - \lfloor x \rfloor \) denote the fractional part of \( x \). Then \( m \in S(\alpha) \) if and only if \( f(m/\alpha) \in (1 - 1/\alpha, 1) \cup \{0\} \). In particular, this means that \( S(\alpha) \cap \{1, \ldots, n\} \) contains \( \lceil (n+1)/\alpha \rceil - 1 \) elements, and similarly. Hence for every integer \( n \),

\[
n = \left\lceil \frac{n+1}{\alpha} \right\rceil + \left\lceil \frac{n+1}{\beta} \right\rceil + \left\lceil \frac{n+1}{\gamma} \right\rceil - 3.
\]

Dividing through by \( n \) and taking the limit as \( n \to \infty \) shows that \( 1/\alpha + 1/\beta + 1/\gamma = 1 \). That in turn implies that for all \( n \),

\[
\left\{ -\frac{n+1}{\alpha} \right\} + \left\{ -\frac{n+1}{\beta} \right\} + \left\{ -\frac{n+1}{\gamma} \right\} = 2.
\]

Our desired contradiction is equivalent to showing that the left side actually takes the value 1 for some \( n \). Since