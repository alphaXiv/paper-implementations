For future use, we notice that changing \( x \) by \( ax \) in the equality of the previous example, we get

\[
e^{ax} = \sum_{n \geq 0} \frac{a^n}{n!} x^n
\]

for every \( x \in \mathbb{R} \).

A variation of the argument used in Example 3.8 allows us to expand \( f(x) = (1 + x)^{\alpha} \) in power series when \( |x| < 1 \) (and for a fixed real number \( \alpha \neq 0 \)).

To this end, given \( \alpha \in \mathbb{R} \) and \( n \in \mathbb{Z} \setminus \{0\} \), we begin by defining the **generalized binomial number** \( \binom{\alpha}{n} \) by letting \( \binom{\alpha}{0} = 1 \) and, for \( n \geq 1 \),

\[
\binom{\alpha}{n} = \frac{\alpha(\alpha - 1)(\alpha - 2) \ldots (\alpha - n + 1)}{n!}.
\]

**Lemma 3.9** *Given \( \alpha \in \mathbb{R} \) and \( n \in \mathbb{N} \), we have:*

(a) \( \binom{\alpha}{n} = \binom{\alpha-1}{n} + \binom{\alpha-1}{n-1} \).
(b) \( \frac{n}{\alpha} \binom{\alpha}{n} = \binom{\alpha-1}{n-1} \), for every \( \alpha \neq 0 \).
(c) \( \left| \binom{\alpha}{n} \right| \leq 1 \) whenever \( |\alpha| \leq 1 \).

*Proof*

(a) Is an easy computation:

\[
\begin{align*}
\binom{\alpha}{n} - \binom{\alpha-1}{n} &= \frac{1}{n!} \alpha(\alpha - 1)(\alpha - 2) \ldots (\alpha - n + 1) \\
&\quad - \frac{1}{n!} (\alpha - 1)(\alpha - 2) \ldots (\alpha - n) \\
&= \frac{1}{n!} (\alpha - 1)(\alpha - 2) \ldots (\alpha - n + 1)(\alpha - (\alpha - n)) \\
&= \frac{1}{(n-1)!} (\alpha - 1)(\alpha - 2) \ldots (\alpha - n + 1) \\
&= \binom{\alpha-1}{n-1}.
\end{align*}
\]

(b) Follows immediately from (3.9).

(c) If \( |\alpha| \leq 1 \), it follows from (3.9) and the triangle inequality that

\[
\left| \binom{\alpha}{n} \right| \leq \frac{|\alpha|(|\alpha| + 1)(|\alpha| + 2) \ldots (|\alpha| + n - 1)}{n!} \leq \frac{1 \cdot 2 \cdot \ldots \cdot n}{n!} = 1.
\]