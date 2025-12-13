Solutions to the 68th William Lowell Putnam Mathematical Competition
Saturday, December 1, 2007

Manjul Bhargava, Kiran Kedlaya, and Lenny Ng

A–1 The only such \( \alpha \) are \( 2/3, 3/2, (13 \pm \sqrt{601})/12 \).

First solution: Let \( C_1 \) and \( C_2 \) be the curves \( y = \alpha x^2 + \alpha x + \frac{1}{24} \) and \( x = \alpha y^2 + \alpha y + \frac{1}{24} \), respectively, and let \( L \) be the line \( y = x \). We consider three cases.

If \( C_1 \) is tangent to \( L \), then the point of tangency \( (x, x) \) satisfies
\[
2\alpha x + \alpha = 1, \qquad x = \alpha x^2 + \alpha x + \frac{1}{24};
\]
by symmetry, \( C_2 \) is tangent to \( L \) there, so \( C_1 \) and \( C_2 \) are tangent. Writing \( \alpha = 1/(2x + 1) \) in the first equation and substituting into the second, we must have
\[
x = \frac{x^2 + x}{2x + 1} + \frac{1}{24},
\]
which simplifies to \( 0 = 24x^2 - 2x - 1 = (6x + 1)(4x - 1) \), or \( x \in \{1/4, -1/6\} \). This yields \( \alpha = 1/(2x + 1) \in \{2/3, 3/2\} \).

If \( C_1 \) does not intersect \( L \), then \( C_1 \) and \( C_2 \) are separated by \( L \) and so cannot be tangent.

If \( C_1 \) intersects \( L \) in two distinct points \( P_1, P_2 \), then it is not tangent to \( L \) at either point. Suppose at one of these points, say \( P_1 \), the tangent to \( C_1 \) is perpendicular to \( L \); then by symmetry, the same will be true of \( C_2 \), so \( C_1 \) and \( C_2 \) will be tangent at \( P_1 \). In this case, the point \( P_1 = (x, x) \) satisfies
\[
2\alpha x + \alpha = -1, \qquad x = \alpha x^2 + \alpha x + \frac{1}{24};
\]
writing \( \alpha = -1/(2x + 1) \) in the first equation and substituting into the second, we have
\[
x = -\frac{x^2 + x}{2x + 1} + \frac{1}{24},
\]
or \( x = (-23 \pm \sqrt{601})/72 \). This yields \( \alpha = -1/(2x + 1) = (13 \pm \sqrt{601})/12 \).

If instead the tangents to \( C_1 \) at \( P_1, P_2 \) are not perpendicular to \( L \), then we claim there cannot be any point where \( C_1 \) and \( C_2 \) are tangent. Indeed, if we count intersections of \( C_1 \) and \( C_2 \) (by using \( C_1 \) to substitute for \( y \) in \( C_2 \), then solving for \( y \)), we get at most four solutions counting multiplicity. Two of these are \( P_1 \) and \( P_2 \), and any point of tangency counts for two more. However, off of \( L \), any point of tangency would have a mirror image which is also a point of tangency, and there cannot be six solutions. Hence we have now found all possible \( \alpha \).

Second solution: For any nonzero value of \( \alpha \), the two conics will intersect in four points in the complex projective plane \( \mathbb{P}^2(\mathbb{C}) \). To determine the \( y \)-coordinates of these intersection points, subtract the two equations to obtain
\[
(y - x) = \alpha(x - y)(x + y) + \alpha(x - y).
\]
Therefore, at a point of intersection we have either \( x = y \), or \( x = -1/\alpha - (y + 1) \). Substituting these two possible linear conditions into the second equation shows that the \( y \)-coordinate of a point of intersection is a root of either \( Q_1(y) = \alpha y^2 + (\alpha - 1)y + 1/24 \) or \( Q_2(y) = \alpha y^2 + (\alpha + 1)y + 25/24 + 1/\alpha \).

If two curves are tangent, then the \( y \)-coordinates of at least two of the intersection points will coincide; the converse is also true because one of the curves is the graph of a function in \( x \). The coincidence occurs precisely when either the discriminant of at least one of \( Q_1 \) or \( Q_2 \) is zero, or there is a common root of \( Q_1 \) and \( Q_2 \). Computing the discriminants of \( Q_1 \) and \( Q_2 \) yields (up to constant factors) \( f_1(\alpha) = 6\alpha^2 - 13\alpha + 6 \) and \( f_2(\alpha) = 6\alpha^2 - 13\alpha - 18 \), respectively. If on the other hand \( Q_1 \) and \( Q_2 \) have a common root, it must be also a root of \( Q_2(y) - Q_1(y) = 2y + 1 + 1/\alpha \), yielding \( y = -(1 + \alpha)/(2\alpha) \) and \( 0 = Q_1(y) = -f_2(\alpha)/(24\alpha) \).

Thus the values of \( \alpha \) for which the two curves are tangent must be contained in the set of zeros of \( f_1 \) and \( f_2 \), namely \( 2/3, 3/2, \) and \( (13 \pm \sqrt{601})/12 \).

Remark: The fact that the two conics in \( \mathbb{P}^2(\mathbb{C}) \) meet in four points, counted with multiplicities, is a special case of Bézout’s theorem: two curves in \( \mathbb{P}^2(\mathbb{C}) \) of degrees \( m, n \) and not sharing any common component meet in exactly \( mn \) points when counted with multiplicity.

Many solvers were surprised that the proposers chose the parameter \( 1/24 \) to give two rational roots and two nonrational roots. In fact, they had no choice in the matter: attempting to make all four roots rational by replacing \( 1/24 \) by \( \beta \) amounts to asking for \( \beta^2 + \beta \) and \( \beta^2 + \beta + 1 \) to be perfect squares. This cannot happen outside of trivial cases (\( \beta = 0, -1 \)) ultimately because the elliptic curve 24A1 (in Cremona’s notation) over \( \mathbb{Q} \) has rank 0. (Thanks to Noam Elkies for providing this computation.)

However, there are choices that make the radical milder, e.g., \( \beta = 1/3 \) gives \( \beta^2 + \beta = 4/9 \) and \( \beta^2 + \beta + 1 = 13/9 \), while \( \beta = 3/5 \) gives \( \beta^2 + \beta = 24/25 \) and \( \beta^2 + \beta + 1 = 49/25 \).

A–2 The minimum is 4, achieved by the square with vertices \( (\pm 1, \pm 1) \).