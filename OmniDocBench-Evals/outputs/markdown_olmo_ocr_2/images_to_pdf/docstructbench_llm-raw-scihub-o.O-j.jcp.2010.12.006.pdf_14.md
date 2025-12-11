Fig. 4. The dispersion contours with stepsizes \( \Delta t = 0.01, \Delta = 0.1 \) for Maxwell's equations (46) from (a) exact dispersion; (b) boxscheme; (c) symplectic method and (d) Yee's method. The constant contour values are \( \omega \in [2,4,6,\ldots,24] \).

\[
\varphi = \tan^{-1} \left( \frac{(v_g)_y}{(v_g)_x} \right), \quad |v_g| = \sqrt{(v_g)_x^2 + (v_g)_y^2}.
\]
(48)

Substituting into (48) the vectors \( \kappa \) and \( v_x \) in polar coordinates (44), and let \( a = |\kappa|\Delta \), this yields the propagation angle \( \varphi \) and the propagation speed \( |v_g| \) in terms of \( a \) and \( \theta \).

For example, \( \varphi \) for the boxscheme is given by

\[
\varphi = \tan^{-1} \left( \frac{\sin \left( \frac{1}{2} \sin(\theta)a \right) \cos^3 \left( \frac{1}{2} \cos(\theta)a \right)}{\cos^3 \left( \frac{1}{2} \sin(\theta)a \right) \sin \left( \frac{1}{2} \cos(\theta)a \right)} \right).
\]

Taking the Taylor expansion of this expression with respect to \( a = 0 \) yields,

\[
\varphi \approx \theta - \frac{1}{12} \sin(4\theta)a^2 + O(a^3).
\]
(49)

Similarly, the Taylor expansion of \( |v_g| \) at \( a = 0 \) yields,

\[
|v_g| \approx 1 + \left( \frac{1}{16} \cos(4\theta) - \frac{r^2}{4} + \frac{3}{16} \right)a^2 + O(a^4),
\]
(50)