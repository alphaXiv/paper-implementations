The 68th William Lowell Putnam Mathematical Competition
Saturday, December 1, 2007

A–1 Find all values of \( \alpha \) for which the curves \( y = \alpha x^2 + \alpha x + \frac{1}{24} \) and \( x = \alpha y^2 + \alpha y + \frac{1}{24} \) are tangent to each other.

A–2 Find the least possible area of a convex set in the plane that intersects both branches of the hyperbola \( xy = 1 \) and both branches of the hyperbola \( xy = -1 \). (A set \( S \) in the plane is called *convex* if for any two points in \( S \) the line segment connecting them is contained in \( S \).)

A–3 Let \( k \) be a positive integer. Suppose that the integers \( 1, 2, 3, \ldots, 3k+1 \) are written down in random order. What is the probability that at no time during this process, the sum of the integers that have been written up to that time is a positive integer divisible by 3? Your answer should be in closed form, but may include factorials.

A–4 A *repunit* is a positive integer whose digits in base 10 are all ones. Find all polynomials \( f \) with real coefficients such that if \( n \) is a repunit, then so is \( f(n) \).

A–5 Suppose that a finite group has exactly \( n \) elements of order \( p \), where \( p \) is a prime. Prove that either \( n = 0 \) or \( p \) divides \( n + 1 \).

A–6 A *triangulation* \( \mathcal{T} \) of a polygon \( P \) is a finite collection of triangles whose union is \( P \), and such that the intersection of any two triangles is either empty, or a shared vertex, or a shared side. Moreover, each side is a side of exactly one triangle in \( \mathcal{T} \). Say that \( \mathcal{T} \) is *admissible* if every internal vertex is shared by 6 or more triangles. For example, [figure omitted.] Prove that there is an integer \( M_n \), depending only on \( n \), such that any admissible triangulation of a polygon \( P \) with \( n \) sides has at most \( M_n \) triangles.

B–1 Let \( f \) be a polynomial with positive integer coefficients. Prove that if \( n \) is a positive integer, then \( f(n) \) divides \( f(f(n) + 1) \) if and only if \( n = 1 \). [Editor's note: one must assume \( f \) is nonconstant.]

B–2 Suppose that \( f : [0, 1] \to \mathbb{R} \) has a continuous derivative and that \( \int_0^1 f(x) dx = 0 \). Prove that for every \( \alpha \in (0, 1) \),
\[
\left| \int_0^\alpha f(x) dx \right| \leq \frac{1}{8} \max_{0 \leq x \leq 1} |f'(x)|.
\]

B–3 Let \( x_0 = 1 \) and for \( n \geq 0 \), let \( x_{n+1} = 3x_n + \lfloor x_n \sqrt{5} \rfloor \). In particular, \( x_1 = 5, x_2 = 26, x_3 = 136, x_4 = 712 \). Find a closed-form expression for \( x_{2007} \). (\( \lfloor a \rfloor \) means the largest integer \( \leq a \).)

B–4 Let \( n \) be a positive integer. Find the number of pairs \( P, Q \) of polynomials with real coefficients such that
\[
(P(X))^2 + (Q(X))^2 = X^{2n} + 1
\]
and \( \deg P > \deg Q \).

B–5 Let \( k \) be a positive integer. Prove that there exist polynomials \( P_0(n), P_1(n), \ldots, P_{k-1}(n) \) (which may depend on \( k \)) such that for any integer \( n \),
\[
\left\lfloor \frac{n}{k} \right\rfloor^k = P_0(n) + P_1(n) \left\lfloor \frac{n}{k} \right\rfloor + \cdots + P_{k-1}(n) \left\lfloor \frac{n}{k} \right\rfloor^{k-1}.
\]
(\( \lfloor a \rfloor \) means the largest integer \( \leq a \).)

B–6 For each positive integer \( n \), let \( f(n) \) be the number of ways to make \( n! \) cents using an unordered collection of coins, each worth \( k! \) cents for some \( k, 1 \leq k \leq n \). Prove that for some constant \( C \), independent of \( n \),
\[
n^{n^2/2 - Cn e^{-n^2/4}} \leq f(n) \leq n^{n^2/2 + Cn e^{-n^2/4}}.
\]