(by logarithmic differentiation) or equivalently,
\[
(1-t^2)f''(t) = f(t)[(n-1-r)(1-t)-r(1+t)] \\
= f(t)[(n-1-2r)-(n-1)t]
\]
and then taking the coefficient of \( t^k \) on both sides:
\[
(k+1)x_{k+2} - (k-1)x_k = \\
(n-1-2r)x_{k+1} - (n-1)x_k.
\]
In particular, the largest such \( c \) is \( n-1 \), and \( x_k = \binom{n-1}{k-1} \) for \( k = 1, 2, \ldots, n \).

Greg Kuperberg has suggested an alternate approach to show directly that \( c = n-1 \) is the largest root, without computing the others. Note that the condition \( x_{n+1} = 0 \) states that \( (x_1, \ldots, x_n) \) is an eigenvector of the matrix
\[
A_{ij} = \left\{ \begin{array}{ll}
i & j = i+1 \\
n-j & j = i-1 \\
0 & \text{otherwise}
\end{array} \right.
\]
with eigenvalue \( c \). By the Perron-Frobenius theorem, \( A \) has a unique eigenvector with positive entries, whose eigenvalue has modulus greater than or equal to that of any other eigenvalue, which proves the claim.

B–1 It is trivial to check that \( \frac{m}{6n} = \frac{\{ m \}}{\{ 6n \}} \leq \frac{\{ m \}}{3n} \) for \( 1 \leq m \leq 2n \), that \( 1-\frac{m}{3n} = \frac{\{ m \}}{3n} \leq \frac{\{ m \}}{6n} \) for \( 2n \leq m \leq 3n \), that \( \frac{m}{6n}-1 = \frac{\{ m \}}{6n} \leq \frac{\{ m \}}{3n} \) for \( 3n \leq m \leq 4n \), and that \( 1-\frac{m}{6n} = \frac{\{ m \}}{6n} \leq \frac{\{ m \}}{3n} \) for \( 4n \leq m \leq 6n \). Therefore the desired sum is
\[
\sum_{m=1}^{2n-1} \frac{m}{6n} + \sum_{m=2n}^{3n-1} \left( 1-\frac{m}{3n} \right) \\
+ \sum_{m=3n}^{4n-1} \left( \frac{m}{3n}-1 \right) + \sum_{m=4n}^{6n-1} \left( 1-\frac{m}{6n} \right) = n.
\]

B–2 It suffices to show that \( |f(x)| \) is bounded for \( x \geq 0 \), since \( f(-x) \) satisfies the same equation as \( f(x) \). But then
\[
\frac{d}{dx} \left( (f(x))^2 + (f'(x))^2 \right) = 2f'(x)(f(x)+f''(x)) \\
= -2xg(x)(f'(x))^2 \leq 0,
\]
so that \( (f(x))^2 \leq (f(0))^2 + (f'(0))^2 \) for \( x \geq 0 \).

B–3 The only such \( n \) are the numbers 1–4, 20–24, 100–104, and 120–124. For the proof let
\[
H_n = \sum_{m=1}^n \frac{1}{m},
\]
and introduce the auxiliary function
\[
I_n = \sum_{1 \leq m < n, (m,5)=1} \frac{1}{m}.
\]
It is immediate (e.g., by induction) that \( I_n \equiv 1, -1, 1, 0, 0 \pmod{5} \) for \( n \equiv 1, 2, 3, 4, 5 \pmod{5} \) respectively, and moreover, we have the equality
\[
H_n = \sum_{m=0}^k \frac{1}{5^m} I_{[n/5^m]},
\]
where \( k = k(n) \) denotes the largest integer such that \( 5^k \leq n \). We wish to determine those \( n \) such that the above sum has nonnegative 5–valuation. (By the 5–valuation of a number \( a \) we mean the largest integer \( v \) such that \( a/5^v \) is an integer.)

If \( [n/5^k] \leq 3 \), then the last term in the above sum has 5–valuation \( -k \), since \( I_1, I_2, I_3 \) each have valuation 0; on the other hand, all other terms must have 5–valuation strictly larger than \( -k \). It follows that \( H_n \) has 5–valuation exactly \( -k \); in particular, \( H_n \) has nonnegative 5–valuation in this case if and only if \( k = 0 \), i.e., \( n = 1, 2, \) or \( 3 \).

Suppose now that \( [n/5^k] = 4 \). Then we must also have \( 20 \leq [n/5^{k-1}] \leq 24 \). The former condition implies that the last term of the above sum is \( I_4/5^k = 1/(12 \cdot 5^{k-2}) \), which has 5–valuation \( -(k-2) \).

It is clear that \( I_{20} \equiv I_{24} \equiv 0 \pmod{25} \); hence if \( [n/5^{k-1}] \) equals 20 or 24, then the second–to–last term of the above sum (if it exists) has valuation at least \( -(k-3) \). The third–to–last term (if it exists) is of the form \( I_r/5^{k-2} \), so that the sum of the last term and the third to last term takes the form \( (I_r+1/12)/5^{k-2} \). Since \( I_r \) can be congruent only to 0,1, or -1 (mod 5), and \( 1/12 \equiv 3 \pmod{5} \), we conclude that the sum of the last term and third–to–last term has valuation \( -(k-2) \), while all other terms have valuation strictly higher. Hence \( H_n \) has nonnegative 5–valuation in this case only when \( k \leq 2 \), leading to the values \( n = 4 \) (arising from \( k = 0 \)), 20,24 (arising from \( k = 1 \) and \( [n/5^{k-1}] = 20 \) and 24 resp.), 101, 102, 103, and 104 (arising from \( k = 2 \), \( [n/5^{k-1}] = 20 \) and 120, 121, 122, 123, and 124 (arising from \( k = 2 \), \( [n/5^{k-1}] = 24 \)).

Finally, suppose \( [n/5^k] = 4 \) and \( [n/5^{k-1}] = 21, 22, \) or 23. Then as before, the first condition implies that the last term of the sum in (*) has valuation \( -(k-2) \), while the second condition implies that the second–to–last term in the same sum has valuation \( -(k-1) \). Hence all terms in the sum (*) have 5–valuation strictly higher than \( -(k-1) \), except for the second–to–last term, and therefore \( H_n \) has 5–valuation \( -(k-1) \) in this case. In particular, \( H_n \) is integral (mod 5) in this case if and only if \( k \leq 1 \), which gives the additional values \( n = 21, 22, \) and 23.

B–4 Let \( s_k = \sum_i (-1)^i a_{k-1,i} \) be the given sum (note that \( a_{k-1,i} \) is nonzero precisely for \( i = 0, \ldots, \lfloor \frac{k}{3} \rfloor \)). Since
\[
a_{m+1,n} = a_{m,n} + a_{m,n-1} + a_{m,n-2},
\]