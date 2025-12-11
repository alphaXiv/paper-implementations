B4 For any continuous real-valued function \( f \) defined on the interval \([0,1]\), let

\[
\mu(f) = \int_0^1 f(x)\, dx, \quad \mathrm{Var}(f) = \int_0^1 (f(x) - \mu(f))^2\, dx,
\]
\[
M(f) = \max_{0 \leq x \leq 1} |f(x)|.
\]

Show that if \( f \) and \( g \) are continuous real-valued functions defined on the interval \([0,1]\), then

\[
\mathrm{Var}(fg) \leq 2\mathrm{Var}(f)M(g)^2 + 2\mathrm{Var}(g)M(f)^2.
\]

B5 Let \( X = \{1,2,\ldots,n\} \), and let \( k \in X \). Show that there are exactly \( k \cdot n^{n-1} \) functions \( f : X \to X \) such that for every \( x \in X \) there is a \( j \geq 0 \) such that \( f^{(j)}(x) \leq k \). [Here \( f^{(j)} \) denotes the \( j \)th iterate of \( f \), so that \( f^{(0)}(x) = x \) and \( f^{(j+1)}(x) = f(f^{(j)}(x)) \).]

B6 Let \( n \geq 1 \) be an odd integer. Alice and Bob play the following game, taking alternating turns, with Alice playing first. The playing area consists of \( n \) spaces, arranged in a line. Initially all spaces are empty. At each turn, a player either

– places a stone in an empty space, or
– removes a stone from a nonempty space \( s \), places a stone in the nearest empty space to the left of \( s \) (if such a space exists), and places a stone in the nearest empty space to the right of \( s \) (if such a space exists).

Furthermore, a move is permitted only if the resulting position has not occurred previously in the game. A player loses if he or she is unable to move. Assuming that both players play optimally throughout the game, what moves may Alice make on her first turn?