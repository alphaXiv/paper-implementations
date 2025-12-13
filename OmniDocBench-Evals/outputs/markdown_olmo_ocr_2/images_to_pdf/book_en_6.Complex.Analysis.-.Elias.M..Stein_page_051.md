2 Cauchy’s Theorem and Its Applications

The solution of a large number of problems can be reduced, in the last analysis, to the evaluation of definite integrals; thus mathematicians have been much occupied with this task... However, among many results obtained, a number were initially discovered by the aid of a type of induction based on the passage from real to imaginary. Often passage of this kind led directly to remarkable results. Nevertheless this part of the theory, as has been observed by Laplace, is subject to various difficulties...

After having reflected on this subject and brought together various results mentioned above, I hope to establish the passage from the real to the imaginary based on a direct and rigorous analysis; my researches have thus led me to the method which is the object of this memoir...

A. L. Cauchy, 1827

In the previous chapter, we discussed several preliminary ideas in complex analysis: open sets in \( \mathbb{C} \), holomorphic functions, and integration along curves. The first remarkable result of the theory exhibits a deep connection between these notions. Loosely stated, Cauchy’s theorem says that if \( f \) is holomorphic in an open set \( \Omega \) and \( \gamma \subset \Omega \) is a closed curve whose interior is also contained in \( \Omega \) then

\[
\int_{\gamma} f(z)\,dz = 0.
\]

Many results that follow, and in particular the calculus of residues, are related in one way or another to this fact.

A precise and general formulation of Cauchy’s theorem requires defining unambiguously the “interior” of a curve, and this is not always an easy task. At this early stage of our study, we shall make use of the device of limiting ourselves to regions whose boundaries are curves that are “toy contours.” As the name suggests, these are closed curves whose visualization is so simple that the notion of their interior will be unam-