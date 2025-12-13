which is less than \( c_i \) but approaches \( c_i \) as \( y \to 0 \). By continuity, for \( i = 2, \ldots, m \), every value in the interval \([c_{i-1}, c_i)\) can be achieved, as can \( c_m = 1 \) by the sequence 0, 1.

To show that all costs \( c \) with \( 1/3 < c \leq 1 \) can be achieved, it now suffices to check that for every \( \varepsilon > 0 \), there exists a sequence with cost at most \( 1/3 + \varepsilon \). For instance, if we take \( x_i = i/m \) for \( i = 0, \ldots, m \), the cost becomes

\[
\frac{1}{m^3}(1^2 + \cdots + m^2) = \frac{(m+1)(2m+1)}{6m^2},
\]

which converges to \( 1/3 \) as \( m \to +\infty \).

Reinterpretation. The cost of jumping along a particular sequence is an upper Riemann sum of the function \( t^2 \). The fact that this function admits a Riemann integral implies that for any \( \varepsilon > 0 \), there exists \( \delta_0 \) such that the cost of the sequence \( x_0, \ldots, x_m \) is at most \( 1/3 + \varepsilon \) as long as \( \max_i |x_i - x_{i-1}| < \varepsilon \). (The computation of the integral using the sequence \( x_i = i/m \) was already known to Archimedes.)

B–3 The answer is \( n = 2^k - 1 \) for some integer \( k \geq 1 \). There is a bijection between mediocre subsets of \( \{1, \ldots, n\} \) and mediocre subsets of \( \{2, \ldots, n+1\} \) given by adding 1 to each element of the subset; thus \( A(n+1) - A(n) \) is the number of mediocre subsets of \( \{1, \ldots, n+1\} \) that contain 1. It follows that \( A(n+2) - 2A(n+1) + A_n = (A(n+2) - A(n+1)) - (A(n+1) - A(n)) \) is the difference between the number of mediocre subsets of \( \{1, \ldots, n+2\} \) containing 1 and the number of mediocre subsets of \( \{1, \ldots, n+1\} \) containing 1. This difference is precisely the number of mediocre subsets of \( \{1, \ldots, n+2\} \) containing both 1 and \( n+2 \), which we term “mediocre subsets containing the endpoints.” Since \( \{1, \ldots, n+2\} \) itself is a mediocre subset of itself containing the endpoints, it suffices to prove that this is the only mediocre subset of \( \{1, \ldots, n+2\} \) containing the endpoints if and only if \( n = 2^k - 1 \) for some \( k \).

If \( n \) is not of the form \( 2^k - 1 \), then we can write \( n+1 = 2^a b \) for odd \( b > 1 \). In this case, the set \( \{1 + mb \mid 0 \leq m \leq 2^a\} \) is a mediocre subset of \( \{1, \ldots, n+2\} \) containing the endpoints: the average of \( 1 + m_1 b \) and \( 1 + m_2 b \), namely \( 1 + \frac{m_1 + m_2}{2} b \), is an integer if and only if \( m_1 + m_2 \) is even, in which case this average lies in the set.

It remains to show that if \( n = 2^k - 1 \), then the only mediocre subset of \( \{1, \ldots, n+2\} \) containing the endpoints is itself. This is readily seen by induction on \( k \). For \( k = 1 \), the statement is obvious. For general \( k \), any mediocre subset \( S \) of \( \{1, \ldots, n+2 = 2^k + 1\} \) containing 1 and \( 2^k + 1 \) must also contain their average, \( 2^{k-1} + 1 \). By the induction assumption, the only mediocre subset of \( \{1, \ldots, 2^{k-1} + 1\} \) containing the endpoints is itself, and so \( S \) must contain all integers between 1 and \( 2^{k-1} + 1 \). Similarly, a mediocre subset of \( \{2^{k-1} + 1, \ldots, 2^k + 1\} \) containing the endpoints gives a mediocre subset of \( \{1, \ldots, 2^{k-1} + 1\} \) containing the endpoints by subtracting \( 2^{k-1} \) from each element. By the induction assumption again, it follows that \( S \) must contain all integers between \( 2^{k-1} + 1 \) and \( 2^k + 1 \). Thus \( S = \{1, \ldots, 2^k + 1\} \) and the induction is complete.

Remark. One can also proceed by checking that a nonempty subset of \( \{1, \ldots, n\} \) is mediocre if and only if it is an arithmetic progression with odd common difference. Given this fact, the number of mediocre subsets of \( \{1, \ldots, n+2\} \) containing the endpoints is seen to be the number of odd factors of \( n+1 \), from which the desired result is evident. (The sequence \( A(n) \) appears as sequence A124197 in the Encyclopedia of Integer Sequences.)

B–4 Any polynomial \( P(x, y) \) of degree at most 2009 can be written uniquely as a sum \( \sum_{i=0}^{2009} P_i(x, y) \) in which \( P_i(x, y) \) is a homogeneous polynomial of degree \( i \). For \( r > 0 \), let \( C_r \) be the path \( (r \cos \theta, r \sin \theta) \) for \( 0 \leq \theta \leq 2\pi \). Put \( \lambda(P_i) = \int_{C_1} P_i \); then for \( r > 0 \),

\[
\int_{C_r} P = \sum_{i=0}^{2009} r^i \lambda(P_i).
\]

For fixed \( P \), the right side is a polynomial in \( r \), which vanishes for all \( r > 0 \) if and only if its coefficients vanish. In other words, \( P \) is balanced if and only if \( \lambda(P_i) = 0 \) for \( i = 0, \ldots, 2009 \).

For \( i \) odd, we have \( P_i(-x, -y) = -P_i(x, y) \). Hence \( \lambda(P_i) = 0 \), e.g., because the contributions to the integral from \( \theta \) and \( \theta + \pi \) cancel.

For \( i \) even, \( \lambda(P_i) \) is a linear function of the coefficients of \( P_i \). This function is not identically zero, e.g., because for \( P_i = (x^2 + y^2)^{i/2} \), the integrand is always positive and so \( \lambda(P_i) > 0 \). The kernel of \( \lambda \) on the space of homogeneous polynomials of degree \( i \) is thus a subspace of codimension 1.

It follows that the dimension of \( V \) is

\[
(1 + \cdots + 2010) - 1005 = (2011 - 1) \times 1005 = 2020050.
\]

B–5 First solution. If \( f(x) \geq x \) for all \( x > 1 \), then the desired conclusion clearly holds. We may thus assume hereafter that there exists \( x_0 > 1 \) for which \( f(x_0) < x_0 \).

Rewrite the original differential equation as

\[
f'(x) = 1 - \frac{x^2 + 1}{x^2} \frac{f(x)^2}{1 + f(x)^2}.
\]

Put \( c_0 = \min\{0, f(x_0) - 1/x_0\} \). For all \( x \geq x_0 \), we have \( f'(x) > -1/x^2 \) and so

\[
f(x) \geq f(x_0) - \int_{x_0}^x dt / t^2 > c_0.
\]

In the other direction, we claim that \( f(x) < x \) for all \( x \geq x_0 \). To see this, suppose the contrary; then by continuity, there is a least \( x \geq x_0 \) for which \( f(x) \geq x \), and