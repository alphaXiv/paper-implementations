(difference of squares). The latter is easily seen (e.g., by AM-GM) to have minimum value 6 (achieved at \( x = 1 \)).

B–2 Consider a triangle as described by the problem; label its vertices \( A, B, C \) so that \( A = (a, b), B \) lies on the \( x \)-axis, and \( C \) lies on the line \( y = x \). Further let \( D = (a, -b) \) be the reflection of \( A \) in the \( x \)-axis, and let \( E = (b, a) \) be the reflection of \( A \) in the line \( y = x \). Then \( AB = DB \) and \( AC = CE \), and so the perimeter of \( ABC \) is \( DB + BC + CE \geq DE = \sqrt{(a-b)^2 + (a+b)^2} = \sqrt{2a^2 + 2b^2} \). It is clear that this lower bound can be achieved: just set \( B \) (resp. \( C \)) to be the intersection between the segment \( DE \) and the \( x \)-axis (resp. line \( x = y \)); thus the minimum perimeter is in fact \( \sqrt{2a^2 + 2b^2} \).

B–3 We use the well-known result that the surface area of the “sphere cap” \( \{ (x, y, z) : x^2 + y^2 + z^2 = 1, z \geq z_0 \} \) is simply \( 2\pi(1 - z_0) \). (This result is easily verified using calculus; we omit the derivation here.) Now the desired surface area is just \( 2\pi \) minus the surface areas of five identical halves of sphere caps; these caps, up to isometry, correspond to \( z_0 \) being the distance from the center of the pentagon to any of its sides, i.e., \( z_0 = \cos \frac{\pi}{5} \). Thus the desired area is \( 2\pi - \frac{5}{2} (2\pi(1 - \cos \frac{\pi}{5})) = 5\pi \cos \frac{\pi}{5} - 3\pi \) (i.e., \( B = \pi/2 \)).

B–4 For convenience, define \( f_{m,n}(i) = \lfloor \frac{i}{m} \rfloor + \lfloor \frac{i}{n} \rfloor \), so that the given sum is \( S(m, n) = \sum_{i=0}^{mn-1} (-1)^{f_{m,n}(i)} \). If \( m \) and \( n \) are both odd, then \( S(m, n) \) is the sum of an odd number of \( \pm 1 \)'s, and thus cannot be zero. Now consider the case where \( m \) and \( n \) have opposite parity. Note that \( \lfloor \frac{i}{m} \rfloor + \lfloor \frac{mn-i}{m} \rfloor = k - 1 \) for all integers \( i, k, m \). Thus \( \lfloor \frac{i}{m} \rfloor + \lfloor \frac{mn-i-1}{m} \rfloor = n - 1 \) and \( \lfloor \frac{i}{n} \rfloor + \lfloor \frac{mn-i-1}{n} \rfloor = m - 1 \); this implies that \( f_{m,n}(i) + f_{m,n}(mn-i-1) = m+n-2 \) is odd, and so \( (-1)^{f_{m,n}(i)} = -(-1)^{f_{m,n}(mn-i-1)} \) for all \( i \). It follows that \( S(m, n) = 0 \) if \( m \) and \( n \) have opposite parity.

Now suppose that \( m = 2k \) and \( n = 2l \) are both even. Then \( \lfloor \frac{j}{2m} \rfloor = \lfloor \frac{2j+1}{2m} \rfloor \) for all \( j \), so \( S \) can be computed as twice the sum over only even indices:

\[
S(2k, 2l) = 2 \sum_{i=0}^{2kl-1} (-1)^{f_{k,l}(i)} = S(k, l)(1 + (-1)^{k+l}).
\]

Thus \( S(2k, 2l) \) vanishes if and only if \( S(k, l) \) vanishes (if \( 1 + (-1)^{k+l} = 0 \), then \( k \) and \( l \) have opposite parity and so \( S(k, l) \) also vanishes).

Piecing our various cases together, we easily deduce that \( S(m, n) = 0 \) if and only if the highest powers of 2 dividing \( m \) and \( n \) are different.

B–5 Write \( N = (10^{1998} - 1)/9 \). Then

\[
\begin{align*}
\sqrt{N} &= \frac{10^{999}}{3} \sqrt{1 - 10^{-1998}} \\
&= \frac{10^{999}}{3} (1 - \frac{1}{2} 10^{-1998} + r),
\end{align*}
\]

where \( r < 10^{-2000} \). Now the digits after the decimal point of \( 10^{999}/3 \) are given by .3333... , while the digits after the decimal point of \( \frac{1}{6} 10^{-999} \) are given by .00000... 1666666... . It follows that the first 1000 digits of \( \sqrt{N} \) are given by .33333...3331; in particular, the thousandth digit is 1.

B–6 First solution: Write \( p(n) = n^3 + an^2 + bn + c \). Note that \( p(n) \) and \( p(n+2) \) have the same parity, and recall that any perfect square is congruent to 0 or 1 (mod 4). Thus if \( p(n) \) and \( p(n+2) \) are perfect squares, they are congruent mod 4. But \( p(n+2) - p(n) = 2n^2 + 2b \) (mod 4), which is not divisible by 4 if \( n \) and \( b \) have opposite parity.

Second solution: We prove more generally that for any polynomial \( P(z) \) with integer coefficients which is not a perfect square, there exists a positive integer \( n \) such that \( P(n) \) is not a perfect square. Of course it suffices to assume \( P(z) \) has no repeated factors, which is to say \( P(z) \) and its derivative \( P'(z) \) are relatively prime.

In particular, if we carry out the Euclidean algorithm on \( P(z) \) and \( P'(z) \) without dividing, we get an integer \( D \) (the discriminant of \( P \)) such that the greatest common divisor of \( P(n) \) and \( P'(n) \) divides \( D \) for any \( n \). Now there exist infinitely many primes \( p \) such that \( p \) divides \( P(n) \) for some \( n \): if there were only finitely many, say, \( p_1, \ldots, p_k \), then for any \( n \) divisible by \( m = P(0)p_1p_2 \cdots p_k \), we have \( P(n) \equiv P(0) \pmod{m} \), that is, \( P(n)/P(0) \) is not divisible by \( p_1, \ldots, p_k \), so must be \( \pm 1 \), but then \( P \) takes some value infinitely many times, contradiction. In particular, we can choose some such \( p \) not dividing \( D \), and choose \( n \) such that \( p \) divides \( P(n) \). Then \( P(n+kp) \equiv P(n) + kpP'(n)(\text{mod } p) \) (write out the Taylor series of the left side); in particular, since \( p \) does not divide \( P'(n) \), we can find some \( k \) such that \( P(n+kp) \) is divisible by \( p \) but not by \( p^2 \), and so is not a perfect square.

Third solution: (from David Rusin, David Savitt, and Richard Stanley independently) Assume that \( n^3 + an^2 + bn + c \) is a square for all \( n > 0 \). For sufficiently large \( n \),

\[
(n^{3/2} + \frac{1}{2}an^{1/2} - 1)^2 < n^3 + an^2 + bn + c
\]
\[
< (n^{3/2} + \frac{1}{2}an^{1/2} + 1)^2;
\]

thus if \( n \) is a large even perfect square, we have \( n^3 + an^2 + bn + c = (n^{3/2} + \frac{1}{2}an^{1/2})^2 \). We conclude this is an equality of polynomials, but the right-hand side is not a perfect square for \( n \) an even non-square, contradiction. (The reader might try generalizing this approach to arbitrary polynomials. A related argument, due to Greg Kuperberg: write \( \sqrt{n^3 + an^2 + bn + c} \) as \( n^{3/2} \) times a power series in \( 1/n \) and take two finite differences to get an expression which tends to 0 as \( n \to \infty \), contradiction.)

Note: in case \( n^3 + an^2 + bn + c \) has no repeated factors, it is a square for only finitely many \( n \), by a theorem