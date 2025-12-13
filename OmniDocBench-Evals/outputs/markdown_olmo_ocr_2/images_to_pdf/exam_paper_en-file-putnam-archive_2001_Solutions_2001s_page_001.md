Solutions to the 62nd William Lowell Putnam Mathematical Competition
Saturday, December 1, 2001

Manjul Bhargava, Kiran Kedlaya, and Lenny Ng

A–1 The hypothesis implies ((b*a)*b)*(b*a) = b for all a, b ∈ S (by replacing a by b*a), and hence a*(b*a) = b for all a, b ∈ S (using (b*a)*b = a).

A–2 Let \( P_n \) denote the desired probability. Then \( P_1 = 1/3 \), and, for \( n > 1 \),

\[
P_n = \left( \frac{2n}{2n+1} \right) P_{n-1} + \left( \frac{1}{2n+1} \right) (1 - P_{n-1})
\]
\[
= \left( \frac{2n-1}{2n+1} \right) P_{n-1} + \frac{1}{2n+1}.
\]

The recurrence yields \( P_2 = 2/5, P_3 = 3/7 \), and by a simple induction, one then checks that for general \( n \) one has \( P_n = n/(2n+1) \).

Note: Richard Stanley points out the following noninductive argument. Put \( f(x) = \prod_{k=1}^n (x+2k)/(2k+1) \); then the coefficient of \( x^i \) in \( f(x) \) is the probability of getting exactly \( i \) heads. Thus the desired number is \( (f(1)-f(-1))/2 \), and both values of \( f \) can be computed directly: \( f(1) = 1 \), and

\[
f(-1) = \frac{1}{3} \times \frac{3}{5} \times \cdots \times \frac{2n-1}{2n+1} = \frac{1}{2n+1}.
\]

A–3 By the quadratic formula, if \( P_m(x) = 0 \), then \( x^2 = m \pm 2\sqrt{m} + 2 \), and hence the four roots of \( P_m \) are given by \( S = \{ \pm \sqrt{m} \pm \sqrt{2} \} \). If \( P_m \) factors into two nonconstant polynomials over the integers, then some subset of \( S \) consisting of one or two elements form the roots of a polynomial with integer coefficients.

First suppose this subset has a single element, say \( \sqrt{m} \pm \sqrt{2} \): this element must be a rational number. Then \( (\sqrt{m} \pm \sqrt{2})^2 = 2 + m \pm 2\sqrt{2m} \) is an integer, so \( m \) is twice a perfect square, say \( m = 2n^2 \). But then \( \sqrt{m} \pm \sqrt{2} = (n \pm 1)\sqrt{2} \) is only rational if \( n = \pm 1 \), i.e., if \( m = 2 \).

Next, suppose that the subset contains two elements; then we can take it to be one of \( \{ \sqrt{m} \pm \sqrt{2} \}, \{ \sqrt{2} \pm \sqrt{m} \} \) or \( \{ \pm (\sqrt{m} + \sqrt{2}) \} \). In all cases, the sum and the product of the elements of the subset must be a rational number. In the first case, this means \( 2\sqrt{m} \in \mathbb{Q} \), so \( m \) is a perfect square. In the second case, we have \( 2\sqrt{2} \in \mathbb{Q} \), contradiction. In the third case, we have \( (\sqrt{m} + \sqrt{2})^2 \in \mathbb{Q} \), or \( m + 2 + 2\sqrt{2m} \in \mathbb{Q} \), which means that \( m \) is twice a perfect square.

We conclude that \( P_m(x) \) factors into two nonconstant polynomials over the integers if and only if \( m \) is either a square or twice a square.

Note: a more sophisticated interpretation of this argument can be given using Galois theory. Namely, if \( m \) is neither a square nor twice a square, then the number fields \( \mathbb{Q}(\sqrt{m}) \) and \( \mathbb{Q}(\sqrt{2}) \) are distinct quadratic fields, so their compositum is a number field of degree 4, whose Galois group acts transitively on \( \{ \pm \sqrt{m} \pm \sqrt{2} \} \). Thus \( P_m \) is irreducible.

A–4 Choose \( r, s, t \) so that \( EC = rBC, FA = sCA, GB = tCB \), and let \( [XYZ] \) denote the area of triangle \( XYZ \). Then \( [ABE] = [AFE] \) since the triangles have the same altitude and base. Also \( [ABE] = (BE/BC)[ABC] = 1 - r \), and \( [ECF] = (EC/BC)(CF/CA)[ABC] = r(1-s) \) (e.g., by the law of sines). Adding this all up yields

\[
1 = [ABE] + [ABF] + [ECF]
\]
\[
= 2(1-r) + r(1-s) = 2 - r - rs
\]

or \( r(1+s) = 1 \). Similarly \( s(1+t) = t(1+r) = 1 \).

Let \( f : [0,\infty) \to [0,\infty) \) be the function given by \( f(x) = 1/(1+x) \); then \( f(f(f(r))) = r \). However, \( f(x) \) is strictly decreasing in \( x \), so \( f(f(x)) \) is increasing and \( f(f(f(x))) \) is decreasing. Thus there is at most one \( x \) such that \( f(f(f(x))) = x \); in fact, since the equation \( f(z) = z \) has a positive root \( z = (-1+\sqrt{5})/2 \), we must have \( r = s = t = z \).

We now compute \( [ABF] = (AF/AC)[ABC] = z, [ABR] = (BR/BF)[ABF] = z/2 \), analogously \( [BCS] = [CAT] = z/2 \), and \( [RST] = [ABC] - [ABR] - [BCS] - [CAT] = |1-3z/2| = \frac{7-3\sqrt{5}}{4} \).

Note: the key relation \( r(1+s) = 1 \) can also be derived by computing using homogeneous coordinates or vectors.

A–5 Suppose \( a^{n+1} - (a+1)^n = 2001 \). Notice that \( a^{n+1} + [(a+1)^n - 1] \) is a multiple of \( a \); thus \( a \) divides \( 2002 = 2 \times 7 \times 11 \times 13 \).

Since 2001 is divisible by 3, we must have \( a \equiv 1 \pmod{3} \), otherwise one of \( a^{n+1} \) and \( (a+1)^n \) is a multiple of 3 and the other is not, so their difference cannot be divisible by 3. Now \( a^{n+1} \equiv 1 \pmod{3} \), so we must have \( (a+1)^n \equiv 1 \pmod{3} \), which forces \( n \) to be even, and in particular at least 2.

If \( a \) is even, then \( a^{n+1} - (a+1)^n \equiv -(a+1)^n \pmod{4} \). Since \( n \) is even, \( -(a+1)^n \equiv -1 \pmod{4} \). Since \( 2001 \equiv 1 \pmod{4} \), this is impossible. Thus \( a \) is odd, and so must divide 1001 = 7 × 11 × 13. Moreover, \( a^{n+1} - (a+1)^n \equiv a \pmod{4} \), so \( a \equiv 1 \pmod{4} \).

Of the divisors of 7 × 11 × 13, those congruent to 1 mod 3 are precisely those not divisible by 11 (since 7 and 13 are both congruent to 1 mod 3). Thus \( a \) divides 7 × 13. Now \( a \equiv 1 \pmod{4} \) is only possible if \( a \) divides 13.