Solutions to the 71st William Lowell Putnam Mathematical Competition
Saturday, December 4, 2010

Kiran Kedlaya and Lenny Ng

A–1 The largest such \( k \) is \( \left\lfloor \frac{n+1}{2} \right\rfloor = \left\lceil \frac{n}{2} \right\rceil \). For \( n \) even, this value is achieved by the partition
\[
\{1, n\}, \{2, n-1\}, \ldots;
\]
for \( n \) odd, it is achieved by the partition
\[
\{n\}, \{1, n-1\}, \{2, n-2\}, \ldots.
\]
One way to see that this is optimal is to note that the common sum can never be less than \( n \), since \( n \) itself belongs to one of the boxes. This implies that \( k \leq (1 + \cdots + n)/n = (n+1)/2 \). Another argument is that if \( k > (n+1)/2 \), then there would have to be two boxes with one number each (by the pigeonhole principle), but such boxes could not have the same sum.

Remark. A much subtler question would be to find the smallest \( k \) (as a function of \( n \)) for which no such arrangement exists.

A–2 The only such functions are those of the form \( f(x) = cx + d \) for some real numbers \( c, d \) (for which the property is obviously satisfied). To see this, suppose that \( f \) has the desired property. Then for any \( x \in \mathbb{R} \),
\[
2f'(x) = f(x+2) - f(x)
= (f(x+2) - f(x+1)) + (f(x+1) - f(x))
= f'(x+1) + f'(x).
\]
Consequently, \( f'(x+1) = f'(x) \).

Define the function \( g : \mathbb{R} \to \mathbb{R} \) by \( g(x) = f(x+1) - f(x) \), and put \( c = g(0), d = f(0) \). For all \( x \in \mathbb{R} \), \( g'(x) = f'(x+1) - f'(x) = 0 \), so \( g(x) = c \) identically, and \( f'(x) = f(x+1) - f(x) = g(x) = c \), so \( f(x) = cx + d \) identically as desired.

A–3 If \( a = b = 0 \), then the desired result holds trivially, so we assume that at least one of \( a, b \) is nonzero. Pick any point \( (a_0, b_0) \in \mathbb{R}^2 \), and let \( L \) be the line given by the parametric equation \( L(t) = (a_0, b_0) + (a, b)t \) for \( t \in \mathbb{R} \). By the chain rule and the given equation, we have \( \frac{d}{dt}(h \circ L) = h \circ L' \). If we write \( f = h \circ L : \mathbb{R} \to \mathbb{R} \), then \( f'(t) = f(t) \) for all \( t \). It follows that \( f(t) = Ce^t \) for some constant \( C \). Since \( |f(t)| \leq M \) for all \( t \), we must have \( C = 0 \). It follows that \( h(a_0, b_0) = 0 \); since \( (a_0, b_0) \) was an arbitrary point, \( h \) is identically 0 over all of \( \mathbb{R}^2 \).

A–4 Put
\[
N = 10^{10^{10^9}} + 10^{10^9} + 10^9 - 1.
\]
Write \( n = 2^m k \) with \( m \) a nonnegative integer and \( k \) a positive odd integer. For any nonnegative integer \( j \),
\[
10^{2^m j} \equiv (-1)^j \pmod{10^{2^m} + 1}.
\]
Since \( 10^n \geq n \geq 2^m \geq m + 1 \), \( 10^n \) is divisible by \( 2^n \) and hence by \( 2^{m+1} \), and similarly \( 10^{10^9} \) is divisible by \( 2^{10^9} \) and hence by \( 2^{m+1} \). It follows that
\[
N \equiv 1 + 1 + (-1) + (-1) \equiv 0 \pmod{10^{2^m} + 1}.
\]
Since \( N \geq 10^{10^9} > 10^9 + 1 \geq 10^{2^m} + 1 \), it follows that \( N \) is composite.

A–5 We start with three lemmas.

Lemma 1. If \( x, y \in G \) are nonzero orthogonal vectors, then \( x * x \) is parallel to \( y \).

Proof. Put \( z = x \times y \neq 0 \), so that \( x, y, \) and \( z = x \times y \) are nonzero and mutually orthogonal. Then \( w = x \times z \neq 0 \), so \( w = x * z \) is nonzero and orthogonal to \( x \) and \( z \). However, if \( (x * x) \times y \neq 0 \), then \( w = x * (x * y) = (x * x) * y = (x * x) \times y \) is also orthogonal to \( y \), a contradiction. \( \square \)

Lemma 2. If \( x \in G \) is nonzero, and there exists \( y \in G \) nonzero and orthogonal to \( x \), then \( x * x = 0 \).

Proof. Lemma 1 implies that \( x * x \) is parallel to both \( y \) and \( x \times y \), so it must be zero. \( \square \)

Lemma 3. If \( x, y \in G \) commute, then \( x \times y = 0 \).

Proof. If \( x \times y \neq 0 \), then \( y \times x \) is nonzero and distinct from \( x \times y \). Consequently, \( x * y = x \times y \) and \( y * x = y \times x \neq x * y \). \( \square \)

We proceed now to the proof. Assume by way of contradiction that there exist \( a, b \in G \) with \( a \times b \neq 0 \). Put \( c = a \times b = a * b \), so that \( a, b, c \) are nonzero and linearly independent. Let \( e \) be the identity element of \( G \). Since \( e \) commutes with \( a, b, c \), by Lemma 3 we have \( e \times a = e \times b = e \times c = 0 \). Since \( a, b, c \) span \( \mathbb{R}^3 \), \( e \times x = 0 \) for all \( x \in \mathbb{R}^3 \), so \( e = 0 \).

Since \( b, c, \) and \( b \times c = b * c \) are nonzero and mutually orthogonal, Lemma 2 implies
\[
b * b = c * c = (b * c) * (b * c) = 0 = e.
\]
Hence \( b * c = c * b \), contradicting Lemma 3 because \( b \times c \neq 0 \). The desired result follows.

A–6 First solution. Note that the hypotheses on \( f \) imply that \( f(x) > 0 \) for all \( x \in [0, +\infty) \), so the integrand is a continuous function of \( f \) and the integral makes sense. Rewrite the integral as
\[
\int_0^\infty \left( 1 - \frac{f(x+1)}{f(x)} \right) dx,
\]