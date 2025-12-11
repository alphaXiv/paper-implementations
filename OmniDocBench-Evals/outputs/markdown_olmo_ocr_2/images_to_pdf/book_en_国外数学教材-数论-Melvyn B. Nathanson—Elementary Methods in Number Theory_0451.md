= \sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2}(d^5\delta - 10d^4u^2 + 10d^3\delta u^2 - 20d^2u^4 \\
+ 5d\delta u^4 - 2u^6)
= \sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2}(d^4(n - 11u^2) + 10d^2u^2(n - 3u^2) \\
+ u^4(5n - 7u^2))
= \left\{ \ell^5 \sum_{j=1}^{\ell} (-1)^{\ell-j}(2j-1) \right\}_{n=\ell^2}
= \{ n^3 \}_{n=\ell^2}

by Exercise 1 in Section 14.3.

The upshot of this analysis is the following three identities:

\[
\sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2}(\delta^4(n - 11u^2) + 40\delta^2u^2(n - 3u^2) \\
+ 16u^4(5n - 7u^2)) = \{ 16n^3 - 40n^2 + 25n \}_{n=\ell^2}; \tag{14.16}
\]
\[
\sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2}((3\delta^2u^2 + 12d^2u^2)(n - 3u^2) + (n - 3u^2)^3 \\
- 12u^2(n - u^2)(n - 3u^2)) = \{ 4n^3 - 3n^2 \}_{n=\ell^2}; \tag{14.17}
\]
\[
\sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2}(d^4(n - 11u^2) + 10d^2u^2(n - 3u^2) \\
+ u^4(5n - 7u^2)) = \{ n^3 \}_{n=\ell^2}. \tag{14.18}
\]

We shall eliminate the terms

\[
\sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2} d^2u^2(n - 3u^2)
\]
and
\[
\sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2} \delta^2u^2(n - 3u^2)
\]
from these equations as follows: Multiply equation (14.18) by 16 and add to equation (14.16), then multiply equation (14.17) by 40/3 and subtract. We obtain

\[
\sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2}(n - 11u^2)(16d^4 + \delta^4) + \sum_{\substack{u^2 + d\delta = n \\ \delta \equiv 1 \pmod{2}}} (-1)^{(\delta-1)/2}