Solutions to the 63rd William Lowell Putnam Mathematical Competition
Saturday, December 7, 2002

Kiran Kedlaya and Lenny Ng

A–1 By differentiating \( P_n(x)/(x^k - 1)^{n+1} \), we find that \( P_{n+1}(x) = (x^k - 1)P'_n(x) - (n + 1)kx^{k-1}P_n(x) \); substituting \( x = 1 \) yields \( P_{n+1}(1) = -(n + 1)kP_n(1) \). Since \( P_0(1) = 1 \), an easy induction gives \( P_n(1) = (-k)^n n! \) for all \( n \geq 0 \).

Note: one can also argue by expanding in Taylor series around 1. Namely, we have
\[
\frac{1}{x^k - 1} = \frac{1}{k(x-1) + \cdots} = \frac{1}{k}(x-1)^{-1} + \cdots,
\]
so
\[
\frac{d^n}{dx^n} \frac{1}{x^k - 1} = \frac{(-1)^n n!}{k(x-1)^{n-1}}
\]
and
\[
P_n(x) = (x^k - 1)^{n+1} \frac{d^n}{dx^n} \frac{1}{x^k - 1}
= (k(x-1) + \cdots)^{n+1} \left( \frac{(-1)^n n!}{k}(x-1)^{-n-1} + \cdots \right)
= (-k)^n n! + \cdots.
\]

A–2 Draw a great circle through two of the points. There are two closed hemispheres with this great circle as boundary, and each of the other three points lies in one of them. By the pigeonhole principle, two of those three points lie in the same hemisphere, and that hemisphere thus contains four of the five given points.

Note: by a similar argument, one can prove that among any \( n + 3 \) points on an \( n \)-dimensional sphere, some \( n + 2 \) of them lie on a closed hemisphere. (One cannot get by with only \( n + 2 \) points: put them at the vertices of a regular simplex.) Namely, any \( n \) of the points lie on a great sphere, which forms the boundary of two hemispheres; of the remaining three points, some two lie in the same hemisphere.

A–3 Note that each of the sets \( \{1\}, \{2\}, \ldots, \{n\} \) has the desired property. Moreover, for each set \( S \) with integer average \( m \) that does not contain \( m \), \( S \cup \{m\} \) also has average \( m \), while for each set \( T \) of more than one element with integer average \( m \) that contains \( m \), \( T \setminus \{m\} \) also has average \( m \). Thus the subsets other than \( \{1\}, \{2\}, \ldots, \{n\} \) can be grouped in pairs, so \( T_n - n \) is even.

A–4 (partly due to David Savitt) Player 0 wins with optimal play. In fact, we prove that Player 1 cannot prevent Player 0 from creating a row of all zeroes, a column of all zeroes, or a \( 2 \times 2 \) submatrix of all zeroes. Each of these forces the determinant of the matrix to be zero.

For \( i, j = 1, 2, 3 \), let \( A_{ij} \) denote the position in row \( i \) and column \( j \). Without loss of generality, we may assume that Player 1’s first move is at \( A_{11} \). Player 0 then plays at \( A_{22} \):

\[
\begin{pmatrix}
1 & * & * \\
* & 0 & * \\
* & * & *
\end{pmatrix}
\]

After Player 1’s second move, at least one of \( A_{23} \) and \( A_{32} \) remains vacant. Without loss of generality, assume \( A_{23} \) remains vacant; Player 0 then plays there.

After Player 1’s third move, Player 0 wins by playing at \( A_{21} \) if that position is unoccupied. So assume instead that Player 1 has played there. Thus of Player 1’s three moves so far, two are at \( A_{11} \) and \( A_{21} \). Hence for \( i \) equal to one of 1 or 3, and for \( j \) equal to one of 2 or 3, the following are both true:

(a) The \( 2 \times 2 \) submatrix formed by rows 2 and \( i \) and by columns 2 and 3 contains two zeroes and two empty positions.

(b) Column \( j \) contains one zero and two empty positions.

Player 0 next plays at \( A_{ij} \). To prevent a zero column, Player 1 must play in column \( j \), upon which Player 0 completes the \( 2 \times 2 \) submatrix in (a) for the win.

Note: one can also solve this problem directly by making a tree of possible play sequences. This tree can be considerably collapsed using symmetries: the symmetry between rows and columns, the invariance of the outcome under reordering of rows or columns, and the fact that the scenario after a sequence of moves does not depend on the order of the moves (sometimes called “transposition invariance”).

Note (due to Paul Cheng): one can reduce Determinant Tic-Tac-Toe to a variant of ordinary tic-tac-toe. Namely, consider a tic-tac-toe grid labeled as follows:

<table>
  <tr>
    <th>A<sub>11</sub></th>
    <th>A<sub>22</sub></th>
    <th>A<sub>33</sub></th>
  </tr>
  <tr>
    <td>A<sub>23</sub></td>
    <td>A<sub>31</sub></td>
    <td>A<sub>12</sub></td>
  </tr>
  <tr>
    <td>A<sub>32</sub></td>
    <td>A<sub>13</sub></td>
    <td>A<sub>21</sub></td>
  </tr>
</table>

Then each term in the expansion of the determinant occurs in a row or column of the grid. Suppose Player 1 first plays in the top left. Player 0 wins by playing first in the top row, and second in the left column. Then there are only one row and column left for Player 1 to threaten, and Player 1 cannot already threaten both on the third move, so Player 0 has time to block both.