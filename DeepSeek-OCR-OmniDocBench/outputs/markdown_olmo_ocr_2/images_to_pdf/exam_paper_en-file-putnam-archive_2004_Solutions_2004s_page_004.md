or in other words

\[
\left(1 - \frac{1}{m+n}\right)^{m+n-1} < \left(1 - \frac{1}{n}\right)^{n-1}.
\]

To show this, we check that the function \( f(x) = (1 - 1/x)^{x-1} \) is strictly decreasing for \( x > 1 \); while this can be achieved using the weighted arithmetic-geometric mean inequality, we give a simple calculus proof instead. The derivative of \( \log f(x) \) is \( \log(1 - 1/x) + 1/x \), so it is enough to check that this is negative for \( x > 1 \). An equivalent statement is that \( \log(1-x) + x < 0 \) for \( 0 < x < 1 \); this in turn holds because the function \( g(x) = \log(1-x) + x \) tends to 0 as \( x \to 0^+ \) and has derivative \( 1 - \frac{1}{1-x} < 0 \) for \( 0 < x < 1 \).

B-3 The answer is \( \{a | a > 2\} \). If \( a > 2 \), then the function \( f(x) = 2a/(a-2) \) has the desired property; both perimeter and area of \( R \) in this case are \( 2a^2/(a-2) \). Now suppose that \( a < 2 \), and let \( f(x) \) be a nonnegative continuous function on \( [0,a] \). Let \( P = (x_0, y_0) \) be a point on the graph of \( f(x) \) with maximal y-coordinate; then the area of \( R \) is at most \( ay_0 \) since it lies below the line \( y = y_0 \). On the other hand, the points \( (0,0) \), \( (a,0) \), and \( P \) divide the boundary of \( R \) into three sections. The length of the section between \( (0,0) \) and \( P \) is at least the distance between \( (0,0) \) and \( P \), which is at least \( y_0 \); the length of the section between \( P \) and \( (a,0) \) is similarly at least \( y_0 \); and the length of the section between \( (0,0) \) and \( (a,0) \) is \( a \). Since \( a \leq 2 \), we have \( 2y_0 + a > ay_0 \) and hence the perimeter of \( R \) is strictly greater than the area of \( R \).

B-4 First solution: Identify the xy-plane with the complex plane \( \mathbb{C} \), so that \( P_k \) is the real number \( k \). If \( z \) is sent to \( z' \) by a counterclockwise rotation by \( \theta \) about \( P_k \), then \( z' - k = e^{i\theta}(z - k) \); hence the rotation \( R_k \) sends \( z \) to \( \zeta z + k(1-\zeta) \), where \( \zeta = e^{2\pi i/n} \). It follows that \( R_1 \) followed by \( R_2 \) sends \( z \) to \( \zeta(\zeta z + (1-\zeta)) + 2(1-\zeta) = \zeta^2 z + (1-\zeta)(\zeta + 2) \), and so forth; an easy induction shows that \( R \) sends \( z \) to

\[
\zeta^n z + (1-\zeta)(\zeta^{n-1} + 2\zeta^{n-2} + \cdots + (n-1)\zeta + n).
\]

Expanding the product \( (1-\zeta)(\zeta^{n-1} + 2\zeta^{n-2} + \cdots + (n-1)\zeta + n) \) yields \( -\zeta^n - \zeta^{n-1} - \cdots - \zeta + n = n \). Thus \( R \) sends \( z \) to \( z + n \); in cartesian coordinates, \( R(x,y) = (x+n, y) \).

Second solution: (by Andy Lutomirski, via Ravi Vakil) Imagine a regular \( n \)-gon of side length 1 placed with its top edge on the x-axis and the left endpoint of that edge at the origin. Then the rotations correspond to rolling this \( n \)-gon along the x-axis; after the \( n \) rotations, it clearly ends up in its original rotation and translated \( n \) units to the right. Hence the whole plane must do so as well.

Third solution: (attribution unknown) Viewing each \( R_k \) as a function of a complex number \( z \) as in the first solution, the function \( R_n \circ R_{n-1} \circ \cdots \circ R_1(z) \) is linear in \( z \) with slope \( \zeta^n = 1 \). It thus equals \( z + T \) for some \( T \in \mathbb{C} \). Since \( f_1(1) = 1 \), we can write \( 1 + T = R_n \circ \cdots \circ R_2(1) \). However, we also have

\[
R_n \circ \cdots \circ R_2(1) = R_{n-1} \circ R_1(0) + 1
\]

by the symmetry in how the \( R_j \) are defined. Hence

\[
R_n(1 + T) = R_n \circ R_1(0) + R_n(1) = T + R_n(1);
\]

that is, \( R_n(T) = T \). Hence \( T = n \), as desired.

B-5 First solution: By taking logarithms, we see that the desired limit is \( \exp(L) \), where

\[
L = \lim_{x \to 1^-} \sum_{n=0}^\infty x^n (\ln(1 + x^{n+1}) - \ln(1 + x^n)).
\]

Now

\[
\begin{align*}
&\sum_{n=0}^N x^n (\ln(1 + x^{n+1}) - \ln(1 + x^n)) \\
&= 1/x \sum_{n=0}^N x^{n+1} \ln(1 + x^{n+1}) - \sum_{n=0}^N x^n \ln(1 + x^n) \\
&= x^N \ln(1 + x^{N+1}) - \ln 2 + (1/x - 1) \sum_{n=1}^N x^n \ln(1 + x^n);
\end{align*}
\]

since \( \lim_{N \to \infty} (x^N \ln(1 + x^{N+1})) = 0 \) for \( 0 < x < 1 \), we conclude that \( L = -\ln 2 + \lim_{x \to 1^-} f(x) \), where

\[
f(x) = (1/x - 1) \sum_{n=1}^\infty x^n \ln(1 + x^n)
= (1/x - 1) \sum_{n=1}^\infty \sum_{m=1}^\infty (-1)^{m+1} x^{n+m}/m.
\]

This final double sum converges absolutely when \( 0 < x < 1 \), since

\[
\sum_{n=1}^\infty \sum_{m=1}^\infty x^{n+m}/m = \sum_{n=1}^\infty x^n (-\ln(1 - x^n))
< \sum_{n=1}^\infty x^n (-\ln(1 - x)),
\]

which converges. (Note that \( -\ln(1 - x) \) and \( -\ln(1 - x^2) \) are positive.) Hence we may interchange the summations in \( f(x) \) to obtain

\[
\begin{align*}
f(x) &= (1/x - 1) \sum_{m=1}^\infty \sum_{n=1}^\infty \frac{(-1)^{m+1} x^{(m+1)n}}{m} \\
&= (1/x - 1) \sum_{m=1}^\infty \frac{(-1)^{m+1}}{m} \left( \frac{x^m (1-x)}{1-x^{m+1}} \right).
\end{align*}
\]

This last sum converges absolutely uniformly in \( x \), so it is legitimate to take limits term by term. Since