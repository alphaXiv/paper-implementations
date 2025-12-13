much cleaner variant of this approach (suggested on AoPS, user henrikjb) is to write

\[
\tan^{-1}(x) = \int_0^x \frac{1}{1+y^2} \, dy
\]

and do a change of variable on the resulting double integral.

A4 The minimum number of tiles is \(mn\). To see that this many are required, label the squares \((i, j)\) with \(1 \leq i \leq 2m-1\) and \(1 \leq j \leq 2n-1\), and for each square with \(i, j\) both odd, color the square red; then no tile can cover more than one red square, and there are \(mn\) red squares.

It remains to show that we can cover any \((2m-1) \times (2n-1)\) rectangle with \(mn\) tiles when \(m, n \geq 4\). First note that we can tile any \(2 \times (2k-1)\) rectangle with \(k \geq 3\) by \(k\) tiles: one of the first type, then \(k-2\) of the second type, and finally one of the first type. Thus if we can cover a \(7 \times 7\) square with 16 tiles, then we can do the general \((2m-1) \times (2n-1)\) rectangle, by decomposing this rectangle into a \(7 \times 7\) square in the lower left corner, along with \(m-4\) (\(2 \times 7\)) rectangles to the right of the square, and \(n-4\) \(((2m-1) \times 2)\) rectangles above, and tiling each of these rectangles separately, for a total of \(16 + 4(m-4) + m(n-4) = mn\) tiles.

To cover the \(7 \times 7\) square, note that the tiling must consist of 15 tiles of the first type and 1 of the second type, and that any \(2 \times 3\) rectangle can be covered using 2 tiles of the first type. We may thus construct a suitable covering by covering all but the center square with eight \(2 \times 3\) rectangles, in such a way that we can attach the center square to one of these rectangles to get a shape that can be covered by two tiles. An example of such a covering, with the remaining \(2 \times 3\) rectangles left intact for visual clarity, is depicted below. (Many other solutions are possible.)

![A diagram showing a 7x7 square divided into smaller rectangles, illustrating a tiling solution](page_109_1042_482_312.png)

A5 First solution: For \(s \in G\) and \(r\) a positive integer, define a *representation of s of length r* to be a sequence of values \(m_1, n_1, \ldots, m_r, n_r \in \{-1, 1\}\) for which

\[
s = g^{m_1} h^{n_1} \cdots g^{m_r} h^{n_r}.
\]

We first check that every \(s \in G\) admits at least one representation of some length; this is equivalent to saying that the set \(S\) of \(s \in G\) which admit representations of some length is equal to \(G\) itself. Since \(S\) is closed under the group operation and \(G\) is finite, \(S\) is also closed under formation of inverses and contains the identity element; that is, \(S\) is a subgroup of \(G\). In particular, \(S\) contains not only \(gh\) but also its inverse \(h^{-1}g^{-1}\); since \(S\) also contains \(g^{-1}h\), we deduce that \(S\) contains \(g^{-2}\). Since \(g\) is of odd order in \(G\), \(g^{-2}\) is also a generator of the cyclic subgroup containing \(g\); it follows that \(g \in S\) and hence \(h \in S\). Since we assumed that \(g, h\) generate \(G\), we now conclude that \(S = G\), as claimed.

To complete the proof, we must now check that for each \(s \in G\), the smallest possible length of a representation of \(s\) cannot exceed \(|G|\). Suppose the contrary, and let

\[
s = g^{m_1} h^{n_1} \cdots g^{m_r} h^{n_r}
\]

be a representation of the smallest possible length. Set

\[
s_i = g^{m_1} h^{n_1} \cdots g^{m_i} h^{n_i} \quad (i = 0, \ldots, r-1),
\]

interpreting \(s_0\) as \(e\); since \(r > |G|\) by hypothesis, by the pigeonhole principle there must exist indices \(0 \leq i < j \leq r-1\) such that \(s_i = s_j\). Then

\[
s = g^{m_1} h^{n_1} \cdots g^{m_i} h^{n_i} g^{m_{j+1}} h^{n_{j+1}} \cdots g^{m_r} h^{n_r}
\]

is another representation of \(s\) of length strictly less than \(r\), a contradiction.

Remark: If one considers \(s_1, \ldots, s_r\) instead of \(s_0, \ldots, s_{r-1}\), then the case \(s = e\) must be handled separately: otherwise, one might end up with a representation of length 0 which is disallowed by the problem statement.

Reinterpretation: Note that the elements \(gh, gh^{-1}, g^{-1}h, g^{-1}h^{-1}\) generate \(gh(g^{-1}h)^{-1} = g^2\) and hence all of \(G\) (again using the hypothesis that \(g\) has odd order, as above). Form the Cayley digraph on the set \(G\), i.e., the directed graph with an edge from \(s_1\) to \(s_2\) whenever \(s_2 = s_1 *\) for \(* \in \{gh, gh^{-1}, g^{-1}h, g^{-1}h^{-1}\}\). Since \(G\) is finite, this digraph is strongly connected: there exists at least one path from any vertex to any other vertex (traveling all edges in the correct direction). The shortest such path cannot repeat any vertices (except the starting and ending vertices in case they coincide), and so has length at most \(|G|\).

Second solution: For \(r\) a positive integer, let \(S_r\) be the set of \(s \in G\) which admit a representation of length at most \(r\) (terminology as in the first solution); obviously \(S_r \subseteq S_{r+1}\). We will show that \(S_r \neq S_{r+1}\) unless \(S_r = G\); this will imply by induction on \(r\) that \(\#S_r \geq \min\{r, |G|\}\) and hence that \(S_r = G\) for some \(r \leq |G|\).