Remark: The second solution amounts to the fact that \( g \), being a polynomial with rational coefficients, is continuous for the 2-adic and 5-adic topologies on \( \mathbb{Q} \). By contrast, the first solution uses the “\( \infty \)-adic” topology, i.e., the usual real topology.

A–5 In all solutions, let \( G \) be a finite group of order \( m \).

First solution: By Lagrange’s theorem, if \( m \) is not divisible by \( p \), then \( n = 0 \). Otherwise, let \( S \) be the set of \( p \)-tuples \( (a_0, \ldots, a_{p-1}) \in G^p \) such that \( a_0 \cdots a_{p-1} = e \); then \( S \) has cardinality \( m^{p-1} \), which is divisible by \( p \). Note that this set is invariant under cyclic permutation, that is, if \( (a_0, \ldots, a_{p-1}) \in S \), then \( (a_1, \ldots, a_{p-1}, a_0) \in S \) also. The fixed points under this operation are the tuples \( (a, \ldots, a) \) with \( a^p = e \); all other tuples can be grouped into orbits under cyclic permutation, each of which has size \( p \). Consequently, the number of \( a \in G \) with \( a^p = e \) is divisible by \( p \); since that number is \( n + 1 \) (only \( e \) has order 1), this proves the claim.

Second solution: (by Anand Deopurkar) Assume that \( n > 0 \), and let \( H \) be any subgroup of \( G \) of order \( p \). Let \( S \) be the set of all elements of \( G \setminus H \) of order dividing \( p \), and let \( H \) act on \( G \) by conjugation. Each orbit has size \( p \) except for those which consist of individual elements \( g \) which commute with \( H \). For each such \( g \), \( g \) and \( H \) generate an elementary abelian subgroup of \( G \) of order \( p^2 \). However, we can group these \( g \) into sets of size \( p^2 - p \) based on which subgroup they generate together with \( H \). Hence the cardinality of \( S \) is divisible by \( p \); adding the \( p - 1 \) nontrivial elements of \( H \) gives \( n \equiv -1 \pmod{p} \) as desired.

Third solution: Let \( S \) be the set of elements in \( G \) having order dividing \( p \), and let \( H \) be an elementary abelian \( p \)-group of maximal order in \( G \). If \( |H| = 1 \), then we are done. So assume \( |H| = p^k \) for some \( k \geq 1 \), and let \( H \) act on \( S \) by conjugation. Let \( T \subset S \) denote the set of fixed points of this action. Then the size of every \( H \)-orbit on \( S \) divides \( p^k \), and so \( |S| \equiv |T| \pmod{p} \). On the other hand, \( H \subset T \), and if \( T \) contained an element not in \( H \), then that would contradict the maximality of \( H \). It follows that \( H = T \), and so \( |S| \equiv |T| = |H| = p^k \equiv 0 \pmod{p} \), i.e., \( |S| = n + 1 \) is a multiple of \( p \).

Remark: This result is a theorem of Cauchy; the first solution above is due to McKay. A more general (and more difficult) result was proved by Frobenius: for any positive integer \( m \), if \( G \) is a finite group of order divisible by \( m \), then the number of elements of \( G \) of order dividing \( m \) is a multiple of \( m \).

A–6 For an admissible triangulation \( \mathcal{T} \), number the vertices of \( P \) consecutively \( v_1, \ldots, v_n \), and let \( a_i \) be the number of edges in \( \mathcal{T} \) emanating from \( v_i \); note that \( a_i \geq 2 \) for all \( i \).

We first claim that \( a_1 + \cdots + a_n \leq 4n - 6 \). Let \( V, E, F \) denote the number of vertices, edges, and faces in \( \mathcal{T} \). By Euler’s Formula, \( (F+1) - E + V = 2 \) (one must add 1 to the face count for the region exterior to \( P \)). Each face has three edges, and each edge but the \( n \) outside edges belongs to two faces; hence \( F = 2E - n \). On the other hand, each edge has two endpoints, and each of the \( V - n \) internal vertices is an endpoint of at least 6 edges; hence \( a_1 + \cdots + a_n + 6(V - n) \leq 2E \). Combining this inequality with the previous two equations gives

\[
a_1 + \cdots + a_n \leq 2E + 6n - 6(1 - F + E) \\
= 4n - 6,
\]

as claimed.

Now set \( A_3 = 1 \) and \( A_n = A_{n-1} + 2n - 3 \) for \( n \geq 4 \); we will prove by induction on \( n \) that \( \mathcal{T} \) has at most \( A_n \) triangles. For \( n = 3 \), since \( a_1 + a_2 + a_3 = 6, a_1 = a_2 = a_3 = 2 \) and hence \( \mathcal{T} \) consists of just one triangle.

Next assume that an admissible triangulation of an \( (n-1) \)-gon has at most \( A_{n-1} \) triangles, and let \( \mathcal{T} \) be an admissible triangulation of an \( n \)-gon. If any \( a_i = 2 \), then we can remove the triangle of \( \mathcal{T} \) containing vertex \( v_i \) to obtain an admissible triangulation of an \( (n-1) \)-gon; then the number of triangles in \( \mathcal{T} \) is at most \( A_{n-1} + 1 < A_n \) by induction. Otherwise, all \( a_i \geq 3 \). Now the average of \( a_1, \ldots, a_n \) is less than 4, and thus there are more \( a_i = 3 \) than \( a_i \geq 5 \). It follows that there is a sequence of \( k \) consecutive vertices in \( P \) whose degrees are 3, 4, 4, \ldots, 4, 3 in order, for some \( k \) with \( 2 \leq k \leq n - 1 \) (possibly \( k = 2 \), in which case there are no degree 4 vertices separating the degree 3 vertices). If we remove from \( \mathcal{T} \) the \( 2k - 1 \) triangles which contain at least one of these vertices, then we are left with an admissible triangulation of an \( (n-1) \)-gon. It follows that there are at most \( A_{n-1} + 2k - 1 \leq A_{n-1} + 2n - 3 = A_n \) triangles in \( \mathcal{T} \). This completes the induction step and the proof.

Remark: We can refine the bound \( A_n \) somewhat. Supposing that \( a_i \geq 3 \) for all \( i \), the fact that \( a_1 + \cdots + a_n \leq 4n - 6 \) implies that there are at least six more indices \( i \) with \( a_i = 3 \) than with \( a_i \geq 5 \). Thus there exist six sequences with degrees 3, 4, \ldots, 4, 3, of total length at most \( n + 6 \). We may thus choose a sequence of length \( k \leq \lfloor \frac{n}{6} \rfloor + 1 \), so we may improve the upper bound to \( A_n = A_{n-1} + 2 \lfloor \frac{n}{6} \rfloor + 1 \), or asymptotically \( \frac{1}{6} n^2 \).

However (as noted by Noam Elkies), a hexagonal swatch of a triangular lattice, with the boundary as close to regular as possible, achieves asymptotically \( \frac{1}{6} n^2 \) triangles.

B–1 The problem fails if \( f \) is allowed to be constant, e.g., take \( f(n) = 1 \). We thus assume that \( f \) is nonconstant. Write \( f(n) = \sum_{i=0}^d a_i n^i \) with \( a_i > 0 \). Then

\[
f(f(n) + 1) = \sum_{i=0}^d a_i (f(n) + 1)^i \\
\equiv f(1) \pmod{f(n)}.
\]

If \( n = 1 \), then this implies that \( f(f(n) + 1) \) is divisible by \( f(n) \). Otherwise, \( 0 < f(1) < f(n) \) since \( f \) is nonconstant and has positive coefficients, so \( f(f(n) + 1) \) cannot be divisible by \( f(n) \).