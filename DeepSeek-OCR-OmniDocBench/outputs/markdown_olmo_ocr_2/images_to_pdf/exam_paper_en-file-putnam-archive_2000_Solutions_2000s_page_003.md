It follows from the definition of \( S_n \), and induction on \( n \), that

\[
\sum_{j \in S_n} x^j \equiv (1 + x) \sum_{j \in S_{n-1}} x^j \\
\equiv (1 + x)^n \sum_{j \in S_0} x^j \pmod{2}.
\]

From the identity \( (x + y)^2 \equiv x^2 + y^2 \pmod{2} \) and induction on \( n \), we have \( (x + y)^{2^n} \equiv x^{2^n} + y^{2^n} \pmod{2} \). Hence if we choose \( N \) to be a power of 2 greater than \( \max\{S_0\} \), then

\[
\sum_{j \in S_n} \equiv (1 + x^N) \sum_{j \in S_0} x^j
\]

and \( S_N = S_0 \cup \{N + a : a \in S_0\} \), as desired.

B–6 For each point \( P \) in \( B \), let \( S_P \) be the set of points with all coordinates equal to \( \pm 1 \) which differ from \( P \) in exactly one coordinate. Since there are more than \( 2^{n+1}/n \) points in \( B \), and each \( S_P \) has \( n \) elements, the cardinalities of the sets \( S_P \) add up to more than \( 2^{n+1} \), which is to say, more than twice the total number of points. By the pigeonhole principle, there must be a point in three of the sets, say \( S_P, S_Q, S_R \). But then any two of \( P, Q, R \) differ in exactly two coordinates, so \( PQR \) is an equilateral triangle, as desired.