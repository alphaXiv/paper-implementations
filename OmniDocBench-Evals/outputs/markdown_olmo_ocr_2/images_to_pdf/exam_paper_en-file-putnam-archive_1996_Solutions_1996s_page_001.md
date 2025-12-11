Solutions to the 57th William Lowell Putnam Mathematical Competition
Saturday, December 7, 1996

Manjul Bhargava and Kiran Kedlaya

A–1 If x and y are the sides of two squares with combined area 1, then \( x^2 + y^2 = 1 \). Suppose without loss of generality that \( x \geq y \). Then the shorter side of a rectangle containing both squares without overlap must be at least x, and the longer side must be at least \( x + y \). Hence the desired value of A is the maximum of \( x(x+y) \).

To find this maximum, we let \( x = \cos \theta, y = \sin \theta \) with \( \theta \in [0, \pi/4] \). Then we are to maximize

\[
\cos^2 \theta + \sin \theta \cos \theta = \frac{1}{2}(1 + \cos 2\theta + \sin 2\theta)
= \frac{1}{2} + \frac{\sqrt{2}}{2} \cos(2\theta - \pi/4)
\leq \frac{1+\sqrt{2}}{2},
\]

with equality for \( \theta = \pi/8 \). Hence this value is the desired value of A.

A–2 Let \( O_1 \) and \( O_2 \) be the centers of \( C_1 \) and \( C_2 \), respectively. (We are assuming \( C_1 \) has radius 1 and \( C_2 \) has radius 3.) Then the desired locus is an annulus centered at the midpoint of \( O_1O_2 \), with inner radius 1 and outer radius 2.

For a fixed point Q on \( C_2 \), the locus of the midpoints of the segments PQ for P lying on \( C_1 \) is the image of \( C_1 \) under a homothety centered at Q of radius 1/2, which is a circle of radius 1/2. As Q varies, the center of this smaller circle traces out a circle \( C_3 \) of radius 3/2 (again by homothety). By considering the two positions of Q on the line of centers of the circles, one sees that \( C_3 \) is centered at the midpoint of \( O_1O_2 \), and the locus is now clearly the specified annulus.

A–3 The claim is false. There are \( \binom{6}{3} = 20 \) ways to choose 3 of the 6 courses; have each student choose a different set of 3 courses. Then each pair of courses is chosen by 4 students (corresponding to the four ways to complete this pair to a set of 3 courses) and is not chosen by 4 students (corresponding to the 3-element subsets of the remaining 4 courses).

Note: Assuming that no two students choose the same courses, the above counterexample is unique (up to permuting students). This may be seen as follows: Given a group of students, suppose that for any pair of courses (among the six) there are at most 4 students taking both, and at most 4 taking neither. Then there are at most 120 = \( (4+4)\binom{6}{2} \) pairs \( (s,p) \), where s is a student, and p is a set of two courses of which s is taking either both or none. On the other hand, if a student s is taking k courses, then he/she occurs in \( f(k) = \binom{k}{2} + \binom{6-k}{2} \) such pairs \( (s,p) \). As \( f(k) \) is minimized for \( k = 3 \), it follows that every student occurs in at least \( 6 = \binom{3}{2} + \binom{3}{2} \) such pairs \( (s,p) \). Hence there can be at most \( 120/6 = 20 \) students, with equality only if each student takes 3 courses, and for each set of two courses, there are exactly 4 students who take both and exactly 4 who take neither. Since there are only 4 ways to complete a given pair of courses to a set of 3, and only 4 ways to choose 3 courses not containing the given pair, the only way for there to be 20 students (under our hypotheses) is if all sets of 3 courses are in fact taken. This is the desired conclusion.

However, Robin Chapman has pointed out that the solution is not unique in the problem as stated, because a given selection of courses may be made by more than one student. One alternate solution is to identify the 6 courses with pairs of antipodal vertices of an icosahedron, and have each student pick a different face and choose the three vertices touching that face. In this example, each of 10 selections is made by a pair of students.

A–4 In fact, we will show that such a function g exists with the property that \( (a,b,c) \in S \) if and only if \( g(d) < g(e) < g(f) \) for some cyclic permutation \( (d,e,f) \) of \( (a,b,c) \). We proceed by induction on the number of elements in A. If \( A = \{a,b,c\} \) and \( (a,b,c) \in S \), then choose g with \( g(a) < g(b) < g(c) \), otherwise choose g with \( g(a) > g(b) > g(c) \).

Now let z be an element of A and \( B = A - \{z\} \). Let \( a_1, \ldots, a_n \) be the elements of B labeled such that \( g(a_1) < g(a_2) < \cdots < g(a_n) \). We claim that there exists a unique \( i \in \{1, \ldots, n\} \) such that \( (a_i, z, a_{i+1}) \in S \), where hereafter \( a_{n+1} = a_1 \).

We show existence first. Suppose no such i exists; then for all \( i, k \in \{1, \ldots, n\} \), we have \( (a_{i+k}, z, a_i) \notin S \). This holds by property 1 for \( k = 1 \) and by induction on k in general, noting that

\[
(a_{i+k+1}, z, a_{i+1}), (a_{i+k}, z, a_i) \in S \\
\Rightarrow (a_{i+k}, a_{i+k+1}, z), (z, a_i, a_{i+k}) \in S \\
\Rightarrow (a_{i+k+1}, z, a_i) \in S.
\]

Applying this when \( k = n \), we get \( (a_{i-1}, z, a_i) \in S \), contradicting the fact that \( (a_i, z, a_{i-1}) \in S \). Hence existence follows.

Now we show uniqueness. Suppose \( (a_i, z, a_{i+1}) \in S \); then for any \( j \neq i-1, i, i+1 \), we have \( (a_i, a_{i+1}, a_j), (a_j, a_{j+1}, a_i) \in S \) by the assumption on G.