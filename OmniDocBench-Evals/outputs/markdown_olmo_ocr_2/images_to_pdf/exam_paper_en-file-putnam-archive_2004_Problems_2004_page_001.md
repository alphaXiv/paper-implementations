The 65th William Lowell Putnam Mathematical Competition
Saturday, December 4, 2004

A1 Basketball star Shanille O’Keal’s team statistician keeps track of the number, \( S(N) \), of successful free throws she has made in her first \( N \) attempts of the season. Early in the season, \( S(N) \) was less than 80% of \( N \), but by the end of the season, \( S(N) \) was more than 80% of \( N \). Was there necessarily a moment in between when \( S(N) \) was exactly 80% of \( N \)?

A2 For \( i = 1, 2 \) let \( T_i \) be a triangle with side lengths \( a_i, b_i, c_i \), and area \( A_i \). Suppose that \( a_1 \leq a_2, b_1 \leq b_2, c_1 \leq c_2 \), and that \( T_2 \) is an acute triangle. Does it follow that \( A_1 \leq A_2 \)?

A3 Define a sequence \( \{u_n\}_{n=0}^{\infty} \) by \( u_0 = u_1 = u_2 = 1 \), and thereafter by the condition that
\[
\det \begin{pmatrix}
u_n & u_{n+1} \\
u_{n+2} & u_{n+3}
\end{pmatrix} = n!
\]
for all \( n \geq 0 \). Show that \( u_n \) is an integer for all \( n \). (By convention, \( 0! = 1 \).)

A4 Show that for any positive integer \( n \) there is an integer \( N \) such that the product \( x_1 x_2 \cdots x_n \) can be expressed identically in the form
\[
x_1 x_2 \cdots x_n = \sum_{i=1}^N c_i (a_{i1} x_1 + a_{i2} x_2 + \cdots + a_{in} x_n)^n
\]
where the \( c_i \) are rational numbers and each \( a_{ij} \) is one of the numbers \( -1, 0, 1 \).

A5 An \( m \times n \) checkerboard is colored randomly: each square is independently assigned red or black with probability 1/2. We say that two squares, \( p \) and \( q \), are in the same connected monochromatic region if there is a sequence of squares, all of the same color, starting at \( p \) and ending at \( q \), in which successive squares in the sequence share a common side. Show that the expected number of connected monochromatic regions is greater than \( mn/8 \).

A6 Suppose that \( f(x, y) \) is a continuous real-valued function on the unit square \( 0 \leq x \leq 1, 0 \leq y \leq 1 \). Show that
\[
\int_0^1 \left( \int_0^1 f(x, y) dx \right)^2 dy + \int_0^1 \left( \int_0^1 f(x, y) dy \right)^2 dx 
\leq \left( \int_0^1 \int_0^1 f(x, y) dxdy \right)^2 + \int_0^1 \int_0^1 [f(x, y)]^2 dxdy.
\]

B1 Let \( P(x) = c_n x^n + c_{n-1} x^{n-1} + \cdots + c_0 \) be a polynomial with integer coefficients. Suppose that \( r \) is a rational number such that \( P(r) = 0 \). Show that the \( n \) numbers
\[
c_n r, c_n r^2, c_{n-1} r, c_n r^3 + c_{n-1} r^2 + c_{n-2} r, \ldots, c_n r^n + c_{n-1} r^{n-1} + \cdots + c_1 r
\]
are integers.

B2 Let \( m \) and \( n \) be positive integers. Show that
\[
\frac{(m+n)!}{(m+n)^{m+n}} < \frac{m!}{m^m} \frac{n!}{n^n}.
\]

B3 Determine all real numbers \( a > 0 \) for which there exists a nonnegative continuous function \( f(x) \) defined on \( [0, a] \) with the property that the region
\[
R = \{ (x, y); 0 \leq x \leq a, 0 \leq y \leq f(x) \}
\]
has perimeter \( k \) units and area \( k \) square units for some real number \( k \).

B4 Let \( n \) be a positive integer, \( n \geq 2 \), and put \( \theta = 2\pi/n \). Define points \( P_k = (k, 0) \) in the xy-plane, for \( k = 1, 2, \ldots, n \). Let \( R_k \) be the map that rotates the plane counterclockwise by the angle \( \theta \) about the point \( P_k \). Let \( R \) denote the map obtained by applying, in order, \( R_1 \), then \( R_2 \), \ldots, then \( R_n \). For an arbitrary point \( (x, y) \), find, and simplify, the coordinates of \( R(x, y) \).

B5 Evaluate
\[
\lim_{x \to 1} \prod_{n=0}^{\infty} \left( \frac{1 + x^{n+1}}{1 + x^n} \right)^{x^n}.
\]

B6 Let \( \mathcal{A} \) be a non-empty set of positive integers, and let \( N(x) \) denote the number of elements of \( \mathcal{A} \) not exceeding \( x \). Let \( \mathcal{B} \) denote the set of positive integers \( b \) that can be written in the form \( b = a - a' \) with \( a \in \mathcal{A} \) and \( a' \in \mathcal{A} \). Let \( b_1 < b_2 < \cdots \) be the members of \( \mathcal{B} \), listed in increasing order. Show that if the sequence \( b_{i+1} - b_i \) is unbounded, then
\[
\lim_{x \to \infty} N(x)/x = 0.
\]