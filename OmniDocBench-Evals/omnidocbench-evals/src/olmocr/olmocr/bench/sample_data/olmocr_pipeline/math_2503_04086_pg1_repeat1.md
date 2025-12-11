Proof. Let $S$ be the generating set associated with $D$ as described in Proposition 2.5. By the circulant diagonalization theorem, the spectrum of $G_R(D) = \Gamma(R, S)$ is the multiset $\{\lambda_g\}_{g \in R}$ where

$$\lambda_g = \sum_{s \in S} \zeta_n^{\psi(gs)} = \sum_{i=1}^k \left[ \sum_{s, Rs = I_i} \zeta_n^{\psi(gs)} \right].$$

We remark that by Corollary 2.7, if $s \in R$ such that $Rs = I_i = Rx_i$ then $s$ has a unique representation of the form $s = ux_i$ where $u \in (R/\text{Ann}_R(x_i))^\times$ and $\hat{u}$ is a fixed lift of $u$ to $R^\times$. With this presentation, we can write

$$\sum_{s, Rs = I_i} \zeta_n^{\psi(gs)} = \sum_{u \in (R/\text{Ann}_R(x_i))^\times} \zeta_n^{\psi(gux_i)} = \sum_{u \in (R/\text{Ann}_R(x_i))^\times} \zeta_n^{\psi_xi(gu)} = c(g, R/\text{Ann}_R(x_i)).$$

Here we recall that $\psi_xi$ is the induced linear functional on $R/\text{Ann}_R(x_i)$. We conclude that $\lambda_g = \sum_{i=1}^k c(g, R/\text{Ann}_R(x_i)).$ \hfill $\square$

The following corollary is simple yet important for our future work on perfect state transfers on gcd-graphs.

**Corollary 4.17.** Suppose that $g' = ug$ for some $u \in R^\times$. Then $\lambda_g = \lambda_{g'}$.

**Acknowledgements**

We thank the Department of Mathematics and Computer Science at Lake Forest College for their generous financial support through an Overleaf subscription. We also thank Ján Mináč for his constant encouragement and support.

**References**

1. Reza Akhtar, Megan Boggess, Tiffany Jackson-Henderson, Isidora Jiménez, Rachel Karpman, Amanda Kinzel, and Dan Pritikin, *On the unitary Cayley graph of a finite ring*, Electron. J. Combin. 16 (2009), no. 1, Research Paper 117, 13 pages.
2. Milan Bašić, Aleksandar Ilić, and Aleksandar Stamenković, *Maximal diameter of integral circulant graphs*, Information and Computation 301 (2024), 105208.
3. Maria Chudnovsky, Michal Cizek, Logan Crew, Ján Mináč, Tung T. Nguyen, Sophie Spirkl, and Nguyễn Duy Tấn, *On prime Cayley graphs*, arXiv:2401.06062, to appear in Journal of Combinatorics (2024).
4. Thomas Honold, *Characterization of finite frobenius rings*, Archiv der Mathematik 76 (2001), no. 6, 406–415.
5. Irving Kaplansky, *Elementary divisors and modules*, Transactions of the American Mathematical Society 66 (1949), no. 2, 464–491.
6. Walter Klotz and Torsten Sander, *Some properties of unitary Cayley graphs*, The Electronic Journal of Combinatorics 14 (2007), no. 1, R45, 12 pages.
7. Erich Lamprecht, *Allgemeine theorie der Gaußschen Summen in endlichen kommutativen Ringen*, Mathematische Nachrichten 9 (1953), no. 3, 149–196.
8. Ján Mináč, Tung T Nguyen, and Nguyen Duy Tấn, *Isomorphic gcd-graphs over polynomial rings*, arXiv preprint arXiv:2411.01768 (2024).
9. ______, *On the gcd graphs over polynomial rings*, arXiv preprint arXiv:2409.01929 (2024).