which holds for \(0 \leq x < 1\), we find that

\[
\log \zeta(s) = \log \prod_p \frac{1}{1 - p^{-s}} = \sum_p \log \left( \frac{1}{1 - p^{-s}} \right) = \sum_{p,m} \frac{p^{-ms}}{m}.
\]

Since the double sum converges absolutely, we need not specify the order of summation. See the Note at the end of this chapter. The formula then holds for all \( \operatorname{Re}(s) > 1 \) by analytic continuation. Note that, by Theorem 6.2 in Chapter 3, \( \log \zeta(s) \) is well defined in the simply connected half-plane \( \operatorname{Re}(s) > 1 \), since \( \zeta \) has no zeros there. Finally, it is clear that we have

\[
\sum_{p,m} \frac{p^{-ms}}{m} = \sum_{n=1}^{\infty} c_n n^{-s},
\]

where \( c_n = 1/m \) if \( n = p^m \) and \( c_n = 0 \) otherwise.

The proof of the theorem we shall give depends on a simple trick that is based on the following inequality.

**Lemma 1.4** *If \( \theta \in \mathbb{R} \), then \( 3 + 4 \cos \theta + \cos 2\theta \geq 0 \).*

This follows at once from the simple observation

\[
3 + 4 \cos \theta + \cos 2\theta = 2(1 + \cos \theta)^2.
\]

**Corollary 1.5** *If \( \sigma > 1 \) and \( t \) is real, then*

\[
\log |\zeta^3(\sigma)\zeta^4(\sigma + it)\zeta(\sigma + 2it)| \geq 0.
\]

*Proof.* Let \( s = \sigma + it \) and note that

\[
\operatorname{Re}(n^{-s}) = \operatorname{Re}(e^{-(\sigma+it)\log n}) = e^{-\sigma \log n} \cos(t \log n) = n^{-\sigma} \cos(t \log n).
\]

Therefore,

\[
\begin{align*}
&\log |\zeta^3(\sigma)\zeta^4(\sigma + it)\zeta(\sigma + 2it)| \\
&= 3 \log |\zeta(\sigma)| + 4 \log |\zeta(\sigma + it)| + \log |\zeta(\sigma + 2it)| \\
&= 3 \operatorname{Re}[\log \zeta(\sigma)] + 4 \operatorname{Re}[\log \zeta(\sigma + it)] + \operatorname{Re}[\log \zeta(\sigma + 2it)] \\
&= \sum c_n n^{-\sigma} (3 + 4 \cos \theta_n + \cos 2\theta_n),
\end{align*}
\]

where \( \theta_n = t \log n \). The positivity now follows from Lemma 1.4, and the fact that \( c_n \geq 0 \).