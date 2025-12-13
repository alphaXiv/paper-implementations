To see that \( n \) divides \( a_n \), factor \( n \) as \( 2^k m \), with \( m \) odd. Then note that \( k \leq n \leq n(n-1)/2 \), and that there exists \( i \leq m-1 \) such that \( m \) divides \( 2^i - 1 \), namely \( i = \phi(m) \) (Euler’s totient function: the number of integers in \( \{1, \ldots, m\} \) relatively prime to \( m \)).

B–1 The answer is 1/3. Let \( G \) be the point obtained by reflecting \( C \) about the line \( AB \). Since \( \angle ADC = \frac{\pi-\theta}{2} \), we find that \( \angle BDE = \pi - \theta - \angle ADC = \frac{\pi-\theta}{2} = \angle ADC = \pi - \angle BDC = \pi - \angle BDG \), so that \( E, D, G \) are collinear. Hence

\[
|EF| = \frac{|BE|}{|BC|} = \frac{|BE|}{|BG|} = \frac{\sin(\theta/2)}{\sin(3\theta/2)},
\]

where we have used the law of sines in \( \triangle BDG \). But by l’Hôpital’s Rule,

\[
\lim_{\theta \to 0} \frac{\sin(\theta/2)}{\sin(3\theta/2)} = \lim_{\theta \to 0} \frac{\cos(\theta/2)}{3 \cos(3\theta/2)} = 1/3.
\]

B–2 First solution: Suppose that \( P \) does not have \( n \) distinct roots; then it has a root of multiplicity at least 2, which we may assume is \( x = 0 \) without loss of generality. Let \( x^k \) be the greatest power of \( x \) dividing \( P(x) \), so that \( P(x) = x^k R(x) \) with \( R(0) \neq 0 \); a simple computation yields

\[
P''(x) = (k^2 - k)x^{k-2}R(x) + 2kx^{k-1}R'(x) + x^k R''(x).
\]

Since \( R(0) \neq 0 \) and \( k \geq 2 \), we conclude that the greatest power of \( x \) dividing \( P''(x) \) is \( x^{k-2} \). But \( P(x) = Q(x)P''(x) \), and so \( x^2 \) divides \( Q(x) \). We deduce (since \( Q \) is quadratic) that \( Q(x) \) is a constant \( C \) times \( x^2 \); in fact, \( C = 1/(n(n-1)) \) by inspection of the leading-degree terms of \( P(x) \) and \( P''(x) \).

Now if \( P(x) = \sum_{j=0}^n a_j x^j \), then the relation \( P(x) = Cx^2 P''(x) \) implies that \( a_j = C j(j-1) a_j \) for all \( j \); hence \( a_j = 0 \) for \( j \leq n-1 \), and we conclude that \( P(x) = a_n x^n \), which has all identical roots.

Second solution (by Greg Kuperberg): Let \( f(x) = P''(x)/P(x) = 1/Q(x) \). By hypothesis, \( f \) has at most two poles (counting multiplicity).

Recall that for any complex polynomial \( P \), the roots of \( P' \) lie within the convex hull of \( P \). To show this, it suffices to show that if the roots of \( P \) lie on one side of a line, say on the positive side of the imaginary axis, then \( P' \) has no roots on the other side. That follows because if \( r_1, \ldots, r_n \) are the roots of \( P \),

\[
\frac{P'(z)}{P(z)} = \sum_{i=1}^n \frac{1}{z - r_i}
\]

and if \( z \) has negative real part, so does \( 1/(z - r_i) \) for \( i = 1, \ldots, n \), so the sum is nonzero.

The above argument also carries through if \( z \) lies on the imaginary axis, provided that \( z \) is not equal to a root of \( P \). Thus we also have that no roots of \( P' \) lie on the sides of the convex hull of \( P \), unless they are also roots of \( P \).

From this we conclude that if \( r \) is a root of \( P \) which is a vertex of the convex hull of the roots, and which is not also a root of \( P' \), then \( f \) has a single pole at \( r \) (as \( r \) cannot be a root of \( P'' \)). On the other hand, if \( r \) is a root of \( P \) which is also a root of \( P' \), it is a multiple root, and then \( f \) has a double pole at \( r \).

If \( P \) has roots not all equal, the convex hull of its roots has at least two vertices.

B–3 We first note that

\[
\sum_{m,n>0} x^m y^n = \frac{xy}{(1-x)(1-y)}.
\]

Subtracting \( S \) from this gives two sums, one of which is

\[
\sum_{m>2n+1} x^m y^n = \sum_n y^n \frac{x^{2n+1}}{1-x} = \frac{x^3 y}{(1-x)(1-x^2 y)}
\]

and the other of which sums to \( xy^3/[(1-y)(1-xy^2)] \). Therefore

\[
S(x,y) = \frac{xy}{(1-x)(1-y)} - \frac{x^3 y}{(1-x)(1-x^2 y)} \\
- \frac{xy^3}{(1-y)(1-xy^2)} \\
= \frac{xy(1+x+y+xy-x^2 y^2)}{(1-x^2 y)(1-xy^2)}
\]

and the desired limit is

\[
\lim_{(x,y) \to (1,1)} xy(1+x+y+xy-x^2 y^2) = 3.
\]

B–4 (based on work by Daniel Stronger) We make repeated use of the following fact: if \( f \) is a differentiable function on all of \( \mathbb{R} \), \( \lim_{x \to \infty} f(x) \geq 0 \), and \( f'(x) > 0 \) for all \( x \in \mathbb{R} \), then \( f(x) > 0 \) for all \( x \in \mathbb{R} \). (Proof: if \( f(y) < 0 \) for some \( x \), then \( f(x) < f(y) \) for all \( x < y \) since \( f' > 0 \), but then \( \lim_{x \to -\infty} f(x) \leq f(y) < 0 \).)

From the inequality \( f'''(x) \leq f(x) \) we obtain

\[
f'' f'''(x) \leq f''(x) f(x) < f''(x) f(x) + f'(x)^2
\]

since \( f'(x) \) is positive. Applying the fact to the difference between the right and left sides, we get

\[
\frac{1}{2} (f''(x))^2 < f(x) f'(x).
\]

On the other hand, since \( f(x) \) and \( f'''(x) \) are both positive for all \( x \), we have

\[
2 f'(x) f'''(x) < 2 f'(x) f''(x) + 2 f(x) f'''(x).
\]

Applying the fact to the difference between the sides yields

\[
f'(x)^2 \leq 2 f(x) f''(x).
\]