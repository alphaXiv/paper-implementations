and \( n \). Thus
\[
2S = \sum_m \sum_n \left( \frac{1}{a_m(a_m + a_n)} + \frac{1}{a_n(a_m + a_n)} \right)
= \sum_m \sum_n \frac{1}{a_m a_n}
= \left( \sum_{n=1}^{\infty} \frac{n}{3^n} \right)^2.
\]
But
\[
\sum_{n=1}^{\infty} \frac{n}{3^n} = \frac{3}{4}
\]
since, e.g., it's \( f'(1) \), where
\[
f(x) = \sum_{n=0}^{\infty} \frac{x^n}{3^n} = \frac{3}{3-x},
\]
and we conclude that \( S = 9/32 \).

A–5 First solution: (by Reid Barton) Let \( r_1, \ldots, r_{1999} \) be the roots of \( P \). Draw a disc of radius \( \varepsilon \) around each \( r_i \), where \( \varepsilon < 1/3998 \); this disc covers a subinterval of \([-1/2, 1/2]\) of length at most \( 2\varepsilon \), and so of the 2000 (or fewer) uncovered intervals in \([-1/2, 1/2]\), one, which we call \( I \), has length at least \( \delta = (1 - 3998\varepsilon)/2000 > 0 \).
We will exhibit an explicit lower bound for the integral of \( |P(x)|/|P(0)| \) over this interval, which will yield such a bound for the entire integral.

Note that
\[
\frac{|P(x)|}{|P(0)|} = \prod_{i=1}^{1999} \frac{|x - r_i|}{|r_i|}.
\]
Also note that by construction, \( |x - r_i| \geq \varepsilon \) for each \( x \in I \). If \( |r_i| \leq 1 \), then we have \( \frac{|x - r_i|}{|r_i|} \geq \varepsilon \). If \( |r_i| > 1 \), then
\[
\frac{|x - r_i|}{|r_i|} = |1 - x/r_i| \geq 1 - |x/r_i| \geq 1/2 > \varepsilon.
\]
We conclude that \( \int_I |P(x)/P(0)| dx \geq \delta \varepsilon \), independent of \( P \).

Second solution: It will be a bit more convenient to assume \( P(0) = 1 \) (which we may achieve by rescaling unless \( P(0) = 0 \), in which case there is nothing to prove) and to prove that there exists \( D > 0 \) such that \( \int_{-1}^1 |P(x)| dx \geq D \), or even such that \( \int_0^1 |P(x)| dx \geq D \).

We first reduce to the case where \( P \) has all of its roots in \([0, 1]\). If this is not the case, we can factor \( P(x) \) as \( Q(x)R(x) \), where \( Q \) has all roots in the interval and \( R \) has none. Then \( R \) is either always positive or always negative on \([0, 1]\); assume the former. Let \( k \) be the largest positive real number such that \( R(x) - kx \geq 0 \) on \([0, 1]\); then
\[
\int_{-1}^1 |P(x)| dx = \int_{-1}^1 |Q(x)R(x)| dx
> \int_{-1}^1 |Q(x)(R(x) - kx)| dx,
\]
and \( Q(x)(R(x) - kx) \) has more roots in \([0, 1]\) than does \( P \) (and has the same value at 0). Repeating this argument shows that \( \int_0^1 |P(x)| dx \) is greater than the corresponding integral for some polynomial with all of its roots in \([0, 1]\).

Under this assumption, we have
\[
P(x) = c \prod_{i=1}^{1999} (x - r_i)
\]
for some \( r_i \in (0, 1] \). Since
\[
P(0) = -c \prod r_i = 1,
\]
we have
\[
|c| \geq \prod |r_i^{-1}| \geq 1.
\]
Thus it suffices to prove that if \( Q(x) \) is a monic polynomial of degree 1999 with all of its roots in \([0, 1]\), then \( \int_0^1 |Q(x)| dx \geq D \) for some constant \( D > 0 \). But the integral of \( \int_0^1 \prod_{i=1}^{1999} |x - r_i| dx \) is a continuous function for \( r_i \in [0, 1] \). The product of all of these intervals is compact, so the integral achieves a minimum value for some \( r_i \). This minimum is the desired \( D \).

Third solution (by Abe Kunin): It suffices to prove the stronger inequality
\[
\sup_{x \in [-1, 1]} |P(x)| \leq C \int_{-1}^1 |P(x)| dx
\]
holds for some \( C \). But this follows immediately from the following standard fact: any two norms on a finite-dimensional vector space (here the polynomials of degree at most 1999) are equivalent. (The proof of this statement is also a compactness argument: \( C \) can be taken to be the maximum of the L1-norm divided by the sup norm over the set of polynomials with L1-norm 1.)

Note: combining the first two approaches gives a constructive solution with a constant that is better than that given by the first solution, but is still far from optimal. I don’t know offhand whether it is even known what the optimal constant and/or the polynomials achieving that constant are.

A–6 Rearranging the given equation yields the much more tractable equation
\[
\frac{a_n}{a_{n-1}} = 6 \frac{a_{n-1}}{a_{n-2}} - 8 \frac{a_{n-2}}{a_{n-3}}.
\]
Let \( b_n = a_n/a_{n-1} \); with the initial conditions \( b_2 = 2, b_3 = 12 \), one easily obtains \( b_n = 2^{n-1}(2^{n-2} - 1) \), and so
\[
a_n = 2^{n(n-1)/2} \prod_{i=1}^{n-1} (2^i - 1).
\]