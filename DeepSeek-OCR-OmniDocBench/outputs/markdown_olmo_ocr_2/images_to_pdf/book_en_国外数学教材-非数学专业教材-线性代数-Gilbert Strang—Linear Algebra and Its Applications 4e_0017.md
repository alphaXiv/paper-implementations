Our true goal is to look beyond two or three dimensions into \( n \) dimensions. With \( n \) equations in \( n \) unknowns, there are \( n \) planes in the row picture. There are \( n \) vectors in the column picture, plus a vector \( b \) on the right side. The equations ask for a *linear combination of the \( n \) columns that equals \( b \)*. For certain equations that will be impossible. Paradoxically, the way to understand the good case is to study the bad one. Therefore we look at the geometry exactly when it breaks down, in the **singular case**.

**Row picture**: Intersection of planes      **Column picture**: Combination of columns

The Singular Case

Suppose we are again in three dimensions, and the three planes in the row picture *do not intersect*. What can go wrong? One possibility is that two planes may be parallel. The equations \( 2u + v + w = 5 \) and \( 4u + 2v + 2w = 11 \) are inconsistent—and parallel planes give no solution (Figure 1.5a shows an end view). In two dimensions, parallel lines are the only possibility for breakdown. But three planes in three dimensions can be in trouble without being parallel.

![Four diagrams showing different plane intersections: (a) two parallel planes, (b) no intersection, (c) line of intersection, (d) all planes parallel](page_370_682_1097_180.png)

Figure 1.5: Singular cases: no solution for (a), (b), or (d), an infinity of solutions for (c).

The most common difficulty is shown in Figure 1.5b. From the end view the planes form a triangle. Every pair of planes intersects in a line, and those lines are parallel. The third plane is not parallel to the other planes, but it is parallel to their line of intersection. This corresponds to a singular system with \( b = (2, 5, 6) \):

\[
\begin{align*}
u &+ v &+ w &= 2 \\
2u &+ 3w &= 5 \tag{3} \\
3u &+ v &+ 4w &= 6.
\end{align*}
\]

No solution, as in Figure 1.5b

The first two left sides add up to the third. On the right side that fails: \( 2 + 5 \neq 6 \). Equation 1 plus equation 2 minus equation 3 is the impossible statement \( 0 = 1 \). Thus the equations are **inconsistent**, as Gaussian elimination will systematically discover.