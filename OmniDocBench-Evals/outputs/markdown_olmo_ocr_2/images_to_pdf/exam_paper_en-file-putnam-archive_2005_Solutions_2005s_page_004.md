so

\[
I = \left( \int_0^1 \frac{xdx}{1+x^2} \right) \left( \int_0^1 \frac{1}{1+y^2} \right) = \log(2) \cdot \frac{\pi}{8}.
\]

Remarks: The first two solutions are related by the fact that if \( x = \tan(\theta) \), then \( 1 - x/(1 + x) = \tan(\pi/4 - \theta) \). The strategy of the third solution (introducing a parameter then differentiating it) was a favorite of physics Nobelist (and Putnam Fellow) Richard Feynman. The fifth solution resembles Gauss’s evaluation of \( \int_{-\infty}^{\infty} \exp(-x^2) dx \). Noam Elkies notes that this integral is number 2.491#8 in Gradshteyn and Ryzhik, Table of integrals, series, and products. The Mathematica computer algebra system (version 5.2) successfully computes this integral, but we do not know how.

A–6 First solution: The angle at a vertex \( P \) is acute if and only if all of the other points lie on an open semicircle. We first deduce from this that if there are any two acute angles at all, they must occur consecutively. Suppose the contrary; label the vertices \( Q_1, \ldots, Q_n \) in counterclockwise order (starting anywhere), and suppose that the angles at \( Q_1 \) and \( Q_i \) are acute for some \( i \) with \( 3 \leq i \leq n-1 \). Then the open semicircle starting at \( Q_2 \) and proceeding counterclockwise must contain all of \( Q_3, \ldots, Q_n \), while the open semicircle starting at \( Q_1 \) and proceeding counterclockwise must contain \( Q_{i+1}, \ldots, Q_n, Q_1, \ldots, Q_{i-1} \). Thus two open semicircles cover the entire circle, contradiction.

It follows that if the polygon has at least one acute angle, then it has either one acute angle or two acute angles occurring consecutively. In particular, there is a unique pair of consecutive vertices \( Q_1, Q_2 \) in counterclockwise order for which \( \angle Q_2 \) is acute and \( \angle Q_1 \) is not acute. Then the remaining points all lie in the arc from the antipode of \( Q_1 \) to \( Q_1 \), but \( Q_2 \) cannot lie in the arc, and the remaining points cannot all lie in the arc from the antipode of \( Q_1 \) to the antipode of \( Q_2 \). Given the choice of \( Q_1, Q_2 \), let \( x \) be the measure of the counterclockwise arc from \( Q_1 \) to \( Q_2 \); then the probability that the other points fall into position is \( 2^{-n+2} - x^{n-2} \) if \( x \leq 1/2 \) and 0 otherwise.

Hence the probability that the polygon has at least one acute angle with a given choice of which two points will act as \( Q_1 \) and \( Q_2 \) is

\[
\int_0^{1/2} (2^{-n+2} - x^{n-2}) dx = \frac{n-2}{n-1} 2^{-n+1}.
\]

Since there are \( n(n-1) \) choices for which two points act as \( Q_1 \) and \( Q_2 \), the probability of at least one acute angle is \( n(n-2)2^{-n+1} \).

Second solution: (by Calvin Lin) As in the first solution, we may compute the probability that for a particular one of the points \( Q_1 \), the angle at \( Q_1 \) is not acute but the following angle is, and then multiply by \( n \). Imagine picking the points by first choosing \( Q_1 \), then picking \( n-1 \) pairs of antipodal points and then picking one member of each pair. Let \( R_2, \ldots, R_n \) be the points of the pairs which lie in the semicircle, taken in order away from \( Q_1 \), and let \( S_2, \ldots, S_n \) be the antipodes of these. Then to get the desired situation, we must choose from the pairs to end up with all but one of the \( S_i \), and we cannot take \( R_n \) and the other \( S_i \) or else \( \angle Q_1 \) will be acute. That gives us \( (n-2) \) good choices out of \( 2^{n-1} \); since we could have chosen \( Q_1 \) to be any of the \( n \) points, the probability is again \( n(n-2)2^{-n+1} \).

B–1 Take \( P(x,y) = (y-2x)(y-2x-1) \). To see that this works, first note that if \( m = \lfloor a \rfloor \), then \( 2m \) is an integer less than or equal to \( 2a \), so \( 2m \leq \lfloor 2a \rfloor \). On the other hand, \( m+1 \) is an integer strictly greater than \( a \), so \( 2m+2 \) is an integer strictly greater than \( 2a \), so \( \lfloor 2a \rfloor \leq 2m+1 \).

B–2 By the arithmetic-harmonic mean inequality or the Cauchy-Schwarz inequality,

\[
(k_1 + \cdots + k_n) \left( \frac{1}{k_1} + \cdots + \frac{1}{k_n} \right) \geq n^2.
\]

We must thus have \( 5n-4 \geq n^2 \), so \( n \leq 4 \). Without loss of generality, we may suppose that \( k_1 \leq \cdots \leq k_n \).

If \( n = 1 \), we must have \( k_1 = 1 \), which works. Note that hereafter we cannot have \( k_1 = 1 \).

If \( n = 2 \), we have \( (k_1, k_2) \in \{(2,4), (3,3)\} \), neither of which work.

If \( n = 3 \), we have \( k_1 + k_2 + k_3 = 11 \), so \( 2 \leq k_1 \leq 3 \). Hence

\[
(k_1, k_2, k_3) \in \{(2,2,7), (2,3,6), (2,4,5), (3,3,5), (3,4,4)\},
\]

and only \( (2,3,6) \) works.

If \( n = 4 \), we must have equality in the AM-HM inequality, which only happens when \( k_1 = k_2 = k_3 = k_4 = 4 \). Hence the solutions are \( n = 1 \) and \( k_1 = 1 \), \( n = 3 \) and \( (k_1, k_2, k_3) \) is a permutation of \( (2,3,6) \), and \( n = 4 \) and \( (k_1, k_2, k_3, k_4) = (4,4,4,4) \).

Remark: In the cases \( n = 2, 3 \), Greg Kuperberg suggests the alternate approach of enumerating the solutions of \( 1/k_1 + \cdots + 1/k_n = 1 \) with \( k_1 \leq \cdots \leq k_n \). This is easily done by proceeding in lexicographic order: one obtains \( (2,2) \) for \( n = 2 \), and \( (2,3,6), (2,4,4), (3,3,3) \) for \( n = 3 \), and only \( (2,3,6) \) contributes to the final answer.

B–3 First solution: The functions are precisely \( f(x) = cx^d \) for \( c, d > 0 \) arbitrary except that we must take \( c = 1 \) in case \( d = 1 \). To see that these work, note that \( f'(a/x) = dc(a/x)^{d-1} \) and \( x/f(x) = 1/(cx^{d-1}) \), so the given equation holds if and only if \( dc^2 a^{d-1} = 1 \). If \( d \neq 1 \), we may solve for \( a \) no matter what \( c \) is; if \( d = 1 \), we must have \( c = 1 \). (Thanks to Brad Rodgers for pointing out the \( d = 1 \) restriction.)

To check that these are all solutions, put \( b = \log(a) \) and \( y = \log(a/x) \); rewrite the given equation as

\[
f(e^{b-y}) f'(e^y) = e^{b-y}.
\]