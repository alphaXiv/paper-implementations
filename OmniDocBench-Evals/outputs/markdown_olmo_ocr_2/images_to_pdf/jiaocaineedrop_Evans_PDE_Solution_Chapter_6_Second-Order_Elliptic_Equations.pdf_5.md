10. Proof. We omit (a) since is standard. For (b), if \( u \) attains an interior maximum, then the conclusion follows from strong maximum principle.

If not, then for some \( x^0 \in \partial U, u(x^0) > u(x) \ \forall x \in U \). Then Hopf’s lemma implies \( \frac{\partial u}{\partial \nu}(x^0) > 0 \), which is a contradiction. \( \square \)

Remark 2. A generalization of this problem to mixed boundary conditions is recorded in Gilbarg-Trudinger, Elliptic PDEs of second order, Problem 3.1.

11. Proof. Define
\[
B[u, v] = \int_U \sum_{i,j} a^{ij} u_{x_i} v_{x_j} \, dx \text{ for } u \in H^1(U), v \in H_0^1(U).
\]
By Exercise 5.17, \( \phi(u) \in H^1(U) \). Then, for all \( v \in C_c^\infty(U), v \geq 0 \),
\[
\begin{align*}
B[\phi(u), v] &= \int_U \sum_{i,j} a^{ij} (\phi(u))_{x_i} v_{x_j} \, dx \\
&= \int_U \sum_{i,j} a^{ij} \phi'(u) u_{x_i} v_{x_j} \, dx, \ (\phi'(u) \text{ is bounded since } u \text{ is bounded}) \\
&= \int_U \sum_{i,j} a^{ij} u_{x_i} (\phi'(u)v)_{x_j} - \sum_{i,j} a_{ij} \phi''(u) u_{x_i} u_{x_j} v \, dx \\
&\leq 0 - \int_U \phi''(u)v|Du|^2 \, dx \leq 0, \text{ by convexity of } \phi.
\end{align*}
\]
(We don’t know whether the product of two \( H^1 \) functions is weakly differentiable. This is why we do not take \( v \in H_0^1 \).) Now we complete the proof with the standard density argument. \( \square \)

12. Proof. Given \( u \in C^2(U) \cap C(\overline{U}) \) with \( Lu \leq 0 \) in \( U \) and \( u \leq 0 \) on \( \partial U \). Since \( \overline{U} \) is compact and \( v \in C(\overline{U}), \ v \geq c > 0 \). So \( w := \frac{u}{v} \in C^2(U) \cap C(\overline{U}) \). Brutal computation gives us
\[
\begin{align*}
-a^{ij} w_{x_i x_j} &= \frac{-a^{ij} u_{x_i x_j} v + a^{ij} v_{x_i x_j} u}{v^2} + \frac{a^{ij} v_{x_i} u_{x_j} - a^{ij} u_{x_i} v_{x_j}}{v^2} - a^{ij} \frac{2}{v} v_{x_j} \frac{v_{x_i} u - v u_{x_i}}{v^2} \\
&= \frac{(Lu - b^i u_{x_i} - c u)v + (-Lv + b^i v_{x_i} + c v)u}{v^2} + 0 + a^{ij} \frac{2}{v} v_{x_j} w_{x_i}, \text{ since } a^{ij} = a^{ji}.
\end{align*}
\]
Therefore,
\[
Mw := -a^{ij} w_{x_i x_j} + w_{x_i} [b^i - a^{ij} \frac{2}{v} v_{x_j}] = \frac{Lu}{v} - \frac{uLv}{v^2} \leq 0 \ \text{on } \{x \in \overline{U} : u > 0\} \subseteq U
\]
If \( \{x \in \overline{U} : u > 0\} \) is not empty, Weak maximum principle to the operator \( M \) with bounded coefficients (since \( v \in C^1(\overline{U}) \)) will lead a contradiction that
\[
0 < \max_{\{u > 0\}} w = \max_{\partial\{u > 0\}} w = \frac{0}{v} = 0
\]
Hence \( u \leq 0 \) in \( U \). \( \square \)