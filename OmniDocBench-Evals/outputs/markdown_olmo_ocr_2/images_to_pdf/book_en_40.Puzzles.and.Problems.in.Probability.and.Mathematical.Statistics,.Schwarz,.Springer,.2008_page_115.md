The last two expectations in this expression are known from the basic result about \( \mathbb{E}[\mathbf{U}] \) so that the only really new part concerns the expectation of the product \( \mathbf{U}\mathbf{V} \). Now, by the definition of \( \mathbf{U}, \mathbf{V} \)

\[
\mathbf{U}\mathbf{V} = \exp(\mathbf{X}) \cdot \exp(\mathbf{Y}) = \exp(\mathbf{X} + \mathbf{Y})
\]

Let the rv \( \mathbf{S} = \mathbf{X} + \mathbf{Y} \). Clearly, \( \mathbf{S} \) is normally distributed, with mean \( \mu_s = \mu_x + \mu_y \) and variance \( \sigma_s^2 = \sigma_x^2 + \sigma_y^2 + 2\rho \sigma_x \sigma_y \). It then follows from the basic result about \( \mathbb{E}[\mathbf{U}] \) that

\[
\begin{align*}
\mathbb{E}[\mathbf{U}\mathbf{V}] &= \mathbb{E}[\exp(\mathbf{S})] \\
&= \exp \left( \mu_s + \frac{1}{2} \sigma_s^2 \right)
\end{align*}
\]

Putting together this result — with \( \mu_s, \sigma_s \) expressed in terms of the original parameters as earlier — and the previously known facts about the means and variances of \( \mathbf{U}, \mathbf{V} \), the correlation coefficient is, after slight simplification, found to be

\[
\operatorname{corr}(\mathbf{U}, \mathbf{V}) = \frac{\exp(\rho \sigma_x \sigma_y) - 1}{\sqrt{[\exp(\sigma_x^2) - 1]\ [\exp(\sigma_y^2) - 1]}}
\]

Note that the correlation between \( \mathbf{U} \) and \( \mathbf{V} \) is completely independent of the means of \( \mathbf{X} \) and \( \mathbf{Y} \). As explained earlier, the exponentiation turns the *location* parameters \( \mu_x, \mu_y \) of \( \mathbf{X} \) and \( \mathbf{Y} \) into *scaling factors* of \( \mathbf{U} \) and \( \mathbf{V} \). Because variations of the scaling factors generally do not change linear correlations, the result was to be expected. On the other hand, the standard deviations \( \sigma_x, \sigma_y \) of \( \mathbf{X} \) and \( \mathbf{Y} \) turn into *powers* for \( \mathbf{U} \) and \( \mathbf{V} \), which generally do influence the linear correlation coefficient.

The result about the correlation of \( \mathbf{U} \) and \( \mathbf{V} \) may also be used to verify the properties we already discussed in part c. Note, in particular, that in the equal-variance case \( \sigma_x = \sigma_y = \sigma \) we get

\[
\operatorname{corr}(\mathbf{U}, \mathbf{V}) = \frac{\exp(\rho \sigma^2) - 1}{\exp(\sigma^2) - 1}
\]

The correlation of \( \mathbf{U} \) and \( \mathbf{V} \) is shown in Figure 3.25 as a function of the correlation \( \rho \) of the original rvs \( \mathbf{X} \) and \( \mathbf{Y} \), for three values of \( \sigma \). For \( \rho = +1 \), the two correlations are identical, but for any other value the correlation between \( \mathbf{U} \) and \( \mathbf{V} \) is weaker than that between \( \mathbf{X} \) and \( \mathbf{Y} \).