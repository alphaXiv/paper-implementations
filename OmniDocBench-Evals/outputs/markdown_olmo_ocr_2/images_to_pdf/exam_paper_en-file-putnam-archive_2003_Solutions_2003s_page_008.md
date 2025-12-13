where

\[
C_j = \#\{i < j : \operatorname{sgn}(a_i) = \operatorname{sgn}(a_j)\}
- \#\{i < j : \operatorname{sgn}(a_i) \neq \operatorname{sgn}(a_j)\}.
\]

Consider the partial sum \( P_k = \sum_{j=1}^k C_j \). If exactly \( p_k \) of \( a_1, \ldots, a_k \) are positive, then this sum is equal to

\[
\binom{p_k}{2} + \binom{k - p_k}{2} - \left[ \binom{k}{2} - \binom{p_k}{2} - \binom{k - p_k}{2} \right],
\]

which expands and simplifies to

\[
-2p_k(k - p_k) + \binom{k}{2}.
\]

For \( k \leq 2p \) even, this partial sum would be minimized with \( p_k = \frac{k}{2} \), and would then equal \( -\frac{k}{2} \); for \( k < 2p \) odd, this partial sum would be minimized with \( p_k = \frac{k+1}{2} \), and would then equal \( -\frac{k-1}{2} \). Either way, \( P_k \geq -\lfloor \frac{k}{2} \rfloor \). On the other hand, if \( k > 2p \), then

\[
-2p_k(k - p_k) + \binom{k}{2} \geq -2p(k - p) + \binom{k}{2}
\]

since \( p_k \) is at most \( p \). Define \( Q_k \) to be \( -\lfloor \frac{k}{2} \rfloor \) if \( k \leq 2p \) and \( -2p(k - p) + \binom{k}{2} \) if \( k \geq 2p \), so that \( P_k \geq Q_k \). Note that \( Q_1 = 0 \).

Partial summation gives

\[
\begin{align*}
\sum_{j=1}^n r_j C_j &= r_n P_n + \sum_{j=2}^n (r_{j-1} - r_j) P_{j-1} \\
&\geq r_n Q_n + \sum_{j=2}^n (r_{j-1} - r_j) Q_{j-1} \\
&= \sum_{j=2}^n r_j (Q_j - Q_{j-1}) \\
&= -r_2 - r_4 - \cdots - r_{2p} + \sum_{j=2p+1}^n (j - 1 - 2p) r_j.
\end{align*}
\]

It follows that

\[
\begin{align*}
\sum_{1 \leq i < j \leq n} |a_i + a_j| &= \sum_{i=1}^n (n-i)r_i + \sum_{j=1}^n r_j C_j \\
&\geq \sum_{i=1}^{2p} (n-i - [i \text{ even}]) r_i \\
&\quad + \sum_{i=2p+1}^n (n-1-2p) r_i \\
&= (n-1-2p) \sum_{i=1}^n r_i \\
&\quad + \sum_{i=1}^{2p} (2p+1-i - [i \text{ even}]) r_i \\
&\geq (n-1-2p) \sum_{i=1}^n r_i + p \sum_{i=1}^{2p} r_i \\
&\geq (n-1-2p) \sum_{i=1}^n r_i + \frac{2p}{n} \sum_{i=1}^n r_i,
\end{align*}
\]

as desired. The next-to-last and last inequalities each follow from the monotonicity of the \( r_i \)'s, the former by pairing the \( i \)th term with the \( (2p+1-i) \)th.

Note: Compare the closely related Problem 6 from the 2000 USA Mathematical Olympiad: prove that for any nonnegative real numbers \( a_1, \ldots, a_n, b_1, \ldots, b_n \), one has

\[
\sum_{i,j=1}^n \min\{a_i a_j, b_i b_j\} \leq \sum_{i,j=1}^n \min\{a_i b_j, a_j b_i\}.
\]