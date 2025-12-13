The 70th William Lowell Putnam Mathematical Competition
Saturday, December 5, 2009

A1 Let \( f \) be a real-valued function on the plane such that for every square \( ABCD \) in the plane, \( f(A) + f(B) + f(C) + f(D) = 0 \). Does it follow that \( f(P) = 0 \) for all points \( P \) in the plane?

A2 Functions \( f, g, h \) are differentiable on some open interval around 0 and satisfy the equations and initial conditions

\[
f' = 2f^2gh + \frac{1}{gh}, \quad f(0) = 1,
\]
\[
g' = fg^2h + \frac{4}{fh}, \quad g(0) = 1,
\]
\[
h' = 3fgh^2 + \frac{1}{fg}, \quad h(0) = 1.
\]

Find an explicit formula for \( f(x) \), valid in some open interval around 0.

A3 Let \( d_n \) be the determinant of the \( n \times n \) matrix whose entries, from left to right and then from top to bottom, are \( \cos 1, \cos 2, \ldots, \cos n^2 \). (For example,

\[
d_3 = \begin{vmatrix}
\cos 1 & \cos 2 & \cos 3 \\
\cos 4 & \cos 5 & \cos 6 \\
\cos 7 & \cos 8 & \cos 9
\end{vmatrix}.
\]

The argument of cos is always in radians, not degrees.) Evaluate \( \lim_{n \to \infty} d_n \).

A4 Let \( S \) be a set of rational numbers such that
(a) \( 0 \in S \);
(b) If \( x \in S \) then \( x+1 \in S \) and \( x-1 \in S \); and
(c) If \( x \in S \) and \( x \notin \{0, 1\} \), then \( \frac{1}{x(x-1)} \in S \).

Must \( S \) contain all rational numbers?

A5 Is there a finite abelian group \( G \) such that the product of the orders of all its elements is \( 2^{2009} \)?

A6 Let \( f : [0,1]^2 \to \mathbb{R} \) be a continuous function on the closed unit square such that \( \frac{\partial f}{\partial x} \) and \( \frac{\partial f}{\partial y} \) exist and are continuous on the interior \( (0,1)^2 \). Let \( a = \int_0^1 f(0,y)\,dy \), \( b = \int_0^1 f(1,y)\,dy \), \( c = \int_0^1 f(x,0)\,dx \), \( d = \int_0^1 f(x,1)\,dx \). Prove or disprove: There must be a point \( (x_0, y_0) \) in \( (0,1)^2 \) such that

\[
\frac{\partial f}{\partial x}(x_0, y_0) = b - a \quad \text{and} \quad \frac{\partial f}{\partial y}(x_0, y_0) = d - c.
\]

B1 Show that every positive rational number can be written as a quotient of products of factorials of (not necessarily distinct) primes. For example,

\[
\frac{10}{9} = \frac{2! \cdot 5!}{3! \cdot 3! \cdot 3!}.
\]

B2 A game involves jumping to the right on the real number line. If \( a \) and \( b \) are real numbers and \( b > a \), the cost of jumping from \( a \) to \( b \) is \( b^3 - ab^2 \). For what real numbers \( c \) can one travel from 0 to 1 in a finite number of jumps with total cost exactly \( c \)?

B3 Call a subset \( S \) of \( \{1, 2, \ldots, n\} \) mediocre if it has the following property: Whenever \( a \) and \( b \) are elements of \( S \) whose average is an integer, that average is also an element of \( S \). Let \( A(n) \) be the number of mediocre subsets of \( \{1, 2, \ldots, n\} \). [For instance, every subset of \( \{1, 2, 3\} \) except \( \{1, 3\} \) is mediocre, so \( A(3) = 7 \).] Find all positive integers \( n \) such that \( A(n+2) - 2A(n+1) + A(n) = 1 \).

B4 Say that a polynomial with real coefficients in two variables, \( x, y \), is balanced if the average value of the polynomial on each circle centered at the origin is 0. The balanced polynomials of degree at most 2009 form a vector space \( V \) over \( \mathbb{R} \). Find the dimension of \( V \).

B5 Let \( f : (1, \infty) \to \mathbb{R} \) be a differentiable function such that

\[
f'(x) = \frac{x^2 - f(x)^2}{x^2(f(x)^2 + 1)}
\]

for all \( x > 1 \).

Prove that \( \lim_{x \to \infty} f(x) = \infty \).

B6 Prove that for every positive integer \( n \), there is a sequence of integers \( a_0, a_1, \ldots, a_{2009} \) with \( a_0 = 0 \) and \( a_{2009} = n \) such that each term after \( a_0 \) is either an earlier term plus \( 2^k \) for some nonnegative integer \( k \), or of the form \( b \bmod c \) for some earlier positive terms \( b \) and \( c \). [Here \( b \bmod c \) denotes the remainder when \( b \) is divided by \( c \), so \( 0 \leq (b \bmod c) < c \).]