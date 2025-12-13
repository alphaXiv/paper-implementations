Solutions to the 73rd William Lowell Putnam Mathematical Competition
Saturday, December 1, 2012

Kiran Kedlaya and Lenny Ng

A1 Without loss of generality, assume \( d_1 \leq d_2 \leq \cdots \leq d_{12} \). If \( d_{i+2}^2 < d_i^2 + d_{i+1}^2 \) for some \( i \leq 10 \), then \( d_i, d_{i+1}, d_{i+2} \) are the side lengths of an acute triangle, since in this case \( d_i^2 < d_{i+1}^2 + d_{i+2}^2 \) and \( d_{i+1}^2 < d_i^2 + d_{i+2}^2 \) as well. Thus we may assume \( d_{i+2}^2 \geq d_i^2 + d_{i+1}^2 \) for all \( i \). But then by induction, \( d_i^2 \geq F_i d_1^2 \) for all \( i \), where \( F_i \) is the \( i \)-th Fibonacci number (with \( F_1 = F_2 = 1 \)); \( i = 1 \) is clear, \( i = 2 \) follows from \( d_2 \geq d_1 \), and the induction step follows from the assumed inequality. Setting \( i = 12 \) now gives \( d_{12}^2 \geq 144 d_1^2 \), contradicting \( d_1 > 1 \) and \( d_{12} < 12 \).

Remark. A materially equivalent problem appeared on the 2012 USA Mathematical Olympiad and USA Junior Mathematical Olympiad.

A2 Write \( d \) for \( a * c = b * c \in S \). For some \( e \in S, d * e = a \), and thus for \( f = c * e, a * f = a * c * e = d * e = a \) and \( b * f = b * c * e = d * e = a \). Let \( g \in S \) satisfy \( g * a = b \); then \( b = g * a = g * (a * f) = (g * a) * f = b * f = a \), as desired.

Remark. With slightly more work, one can show that \( S \) forms an abelian group with the operation \( * \).

A3 We will prove that \( f(x) = \sqrt{1 - x^2} \) for all \( x \in [-1, 1] \). Define \( g : (-1, 1) \to \mathbb{R} \) by \( g(x) = f(x)/\sqrt{1 - x^2} \). Plugging \( f(x) = g(x)\sqrt{1 - x^2} \) into equation (i) and simplifying yields
\[
g(x) = g\left( \frac{x^2}{2 - x^2} \right)
\]
for all \( x \in (-1, 1) \). Now fix \( x \in (-1, 1) \) and define a sequence \( \{a_n\}_{n=1}^\infty \) by \( a_1 = x \) and \( a_{n+1} = \frac{a_n^2}{2 - a_n^2} \). Then \( a_n \in (-1, 1) \) and thus \( |a_{n+1}| \leq |a_n|^2 \) for all \( n \). It follows that \( \{|a_n|\} \) is a decreasing sequence with \( |a_n| \leq |x|^n \) for all \( n \), and so \( \lim_{n \to \infty} a_n = 0 \). Since \( g(a_n) = g(x) \) for all \( n \) by (1) and \( g \) is continuous at 0, we conclude that \( g(x) = g(0) = f(0) = 1 \). This holds for all \( x \in (-1, 1) \) and thus for \( x = \pm 1 \) as well by continuity. The result follows.

Remark. As pointed out by Noam Elkies, condition (iii) is unnecessary. However, one can use it to derive a slightly different solution by running the recursion in the opposite direction.

A4 We begin with an easy lemma.

Lemma. Let \( S \) be a finite set of integers with the following property: for all \( a, b, c \in S \) with \( a \leq b \leq c \), we also have \( a + c - b \in S \). Then \( S \) is an arithmetic progression.

Proof. We may assume \( \#S \geq 3 \), as otherwise \( S \) is trivially an arithmetic progression. Let \( a_1, a_2 \) be the smallest and second-smallest elements of \( S \), respectively, and put \( d = a_2 - a_1 \). Let \( m \) be the smallest positive integer such that \( a_1 + md \notin S \). Suppose that there exists an integer \( n \) contained in \( S \) but not in \( \{a_1, a_1 + d, \ldots, a_1 + (m-1)d\} \), and choose the least such \( n \). By the hypothesis applied with \( (a, b, c) = (a_1, a_2, n) \), we see that \( n - d \) also has the property, a contradiction. \( \square \)

We now return to the original problem. By dividing \( B, q, r \) by \( \gcd(q, r) \) if necessary, we may reduce to the case where \( \gcd(q, r) = 1 \). We may assume \( \#S \geq 3 \), as otherwise \( S \) is trivially an arithmetic progression. Let \( a_1, a_2, a_3 \) be any three distinct elements of \( S \), labeled so that \( a_1 < a_2 < a_3 \), and write \( ra_i = b_i + m_i q \) with \( b_i, m_i \in \mathbb{Z} \) and \( b_i \in B \). Note that \( b_1, b_2, b_3 \) must also be distinct, so the differences \( b_2 - b_1, b_3 - b_1, b_3 - b_2 \) are all nonzero; consequently, two of them have the same sign. If \( b_i - b_j \) and \( b_k - b_l \) have the same sign, then we must have
\[
(a_i - a_j)(b_k - b_l) = (b_i - b_j)(a_k - a_l)
\]
because both sides are of the same sign, of absolute value less than \( q \), and congruent to each other modulo \( q \). In other words, the points \( (a_1, b_1), (a_2, b_2), (a_3, b_3) \) in \( \mathbb{R}^2 \) are collinear. It follows that \( a_4 = a_1 + a_3 - a_2 \) also belongs to \( S \) (by taking \( b_4 = b_1 + b_3 - b_2 \)), so \( S \) satisfies the conditions of the lemma. It is therefore an arithmetic progression.

Reinterpretations. One can also interpret this argument geometrically using cross products (suggested by Noam Elkies), or directly in terms of congruences (suggested by Karl Mahlburg).

Remark. The problem phrasing is somewhat confusing: to say that “\( S \) is the intersection of [the interval] \( A \) with an arithmetic progression” is the same thing as saying that “\( S \) is the empty set or an arithmetic progression” unless it is implied that arithmetic progressions are necessarily infinite. Under that interpretation, however, the problem becomes false; for instance, for
\[
q = 5, r = 1, A = [1, 3], B = [0, 2],
\]
we have
\[
T = \{\cdots, 0, 1, 2, 5, 6, 7, \ldots\}, S = \{1, 2\}.
\]

A5 The pairs \( (p, n) \) with the specified property are those pairs with \( n = 1 \), together with the single pair \( (2, 2) \). We first check that these do work. For \( n = 1 \), it is clear that taking \( v = (1) \) and \( M = (0) \) has the desired effect.