= 2 \sum_{1 \leq u \leq \sqrt{x}} \left\lfloor \frac{x}{u} \right\rfloor - \lfloor \sqrt{x} \rfloor^2
= 2 \sum_{1 \leq u \leq \sqrt{x}} \left( \frac{x}{u} - \left\{ \frac{x}{u} \right\} \right) - (\sqrt{x} - \{ \sqrt{x} \})^2
= 2x \sum_{1 \leq u \leq \sqrt{x}} \frac{1}{u} - 2 \sum_{1 \leq u \leq \sqrt{x}} \left\{ \frac{x}{u} \right\} - x + O(\sqrt{x})
= 2x \left( \log \sqrt{x} + \gamma + O \left( \frac{1}{\sqrt{x}} \right) \right) - x + O(\sqrt{x})
= x \log x + (2\gamma - 1)x + O(\sqrt{x}).

This completes the proof. \( \blacksquare \)

Theorem 7.4 For \( x \geq 1 \),

\[
\Delta(x) = \sum_{n \leq x} (\log n - d(n) + 2\gamma) = O \left( x^{1/2} \right).
\]

Proof. By Theorem 7.3 we have

\[
\sum_{n \leq x} d(n) = x \log x + (2\gamma - 1)x + O \left( x^{1/2} \right).
\]

By Theorem 6.4 we have

\[
\sum_{n \leq x} \log n = x \log x - x + O(\log x).
\]

Subtracting the first equation from the second, we obtain

\[
\sum_{n \leq x} (\log n - d(n) + 2\gamma) = O \left( x^{1/2} \right) - 2\gamma \{ x \} + O(\log x) = O \left( x^{1/2} \right).
\]

\( \blacksquare \)

An *ordered factorization* of the positive integer \( n \) into exactly \( \ell \) factors is an \( \ell \)-tuple \( (d_1, \ldots, d_\ell) \) such that \( n = d_1 \cdots d_\ell \). The divisor function \( d(n) \) counts the number of ordered factorizations of \( n \) into exactly two factors, since each factorization \( n = dd' \) is completely determined by the first factor \( d \). For every positive integer \( \ell \), we define the arithmetic function \( d_\ell(n) \) as the number of factorizations of \( n \) into exactly \( \ell \) factors. Then \( d_1(n) = 1 \) and \( d_2(n) = d(n) \) for all \( n \).