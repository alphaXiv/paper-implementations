What is a digital image?

Mathematical definition:
- A 2D (grayscale) digital image is a 2D function defined on a 2D domain (usually rectangular domain):

\[
f : \Omega \to \mathbb{R}
\]

- \( f(x, y) \) is called the brightness/intensity/grey level;
- (x,y) is the spatial coordinates of the image.
- Thus, a 2D digital image looks like this:

\[
f(x, y) = \begin{bmatrix}
f(1, 1) & f(1, 2) & \ldots & f(1, N) \\
f(2, 1) & f(2, 2) & \ldots & f(2, N) \\
\vdots & \vdots & & \vdots \\
f(N, 1) & f(N, 2) & \ldots & f(N, N)
\end{bmatrix}
\]

- Each element in the matrix is called pixel (picture element);
- Usually, \( 0 \leq f(x, y) \leq G - 1 \) and \( (N = 2^n, \ G = 2^m) \)

IMAGE PROCESSING IS RELATED TO LINEAR ALGEBRA!!