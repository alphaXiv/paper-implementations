Similarly,
\[
S_2 = \sum_{c=1}^{\infty} \sum_{b=c+1}^{\infty} \frac{2^{b+c} - 2^{b-c+1}}{3^b 5^c}
\]
\[
= \sum_{c=1}^{\infty} \left( \left( \left( \frac{2}{5} \right)^c - \frac{2}{10^c} \right) \sum_{b=c+1}^{\infty} \left( \frac{2}{3} \right)^b \right)
\]
\[
= \sum_{c=1}^{\infty} \left( \left( \frac{2}{5} \right)^c - \frac{2}{10^c} \right) 3 \left( \frac{2}{3} \right)^{c+1}
\]
\[
= \sum_{c=1}^{\infty} \left( 2 \left( \frac{4}{15} \right)^c - 4 \left( \frac{1}{15} \right)^c \right)
\]
\[
= \frac{34}{77}.
\]

We conclude that \( S = S_1 + S_2 = \frac{17}{21} \).

**Second solution:** Recall that the real numbers \( a, b, c \) form the side lengths of a triangle if and only if
\[
s - a, s - b, s - c > 0 \qquad s = \frac{a + b + c}{2},
\]
and that if we put \( x = 2(s-a), y = 2(s-b), z = 2(s-c) \),
\[
a = \frac{y+z}{2}, b = \frac{z+x}{2}, c = \frac{x+y}{2}.
\]

To generate all *integer* triples \( (a, b, c) \) which form the side lengths of a triangle, we must also assume that \( x, y, z \) are either all even or all odd. We may therefore write the original sum as
\[
\sum_{x,y,z>0\text{ odd}} \frac{2^{(y+z)/2}}{3^{(z+x)/2} 5^{(x+y)/2}} + \sum_{x,y,z>0\text{ even}} \frac{2^{(y+z)/2}}{3^{(z+x)/2} 5^{(x+y)/2}}.
\]

To unify the two sums, we substitute in the first case \( x = 2u+1, y = 2v+1, z = 2w+1 \) and in the second case \( x = 2u+2, y = 2v+2, z = 2w+2 \) to obtain
\[
\sum_{(a,b,c)\in T} \frac{2^a}{3^b 5^c} = \sum_{u,v,w} \frac{2^{v+w}}{3^{w+u} 5^{u+v}} \left( 1 + \frac{2^{-1}}{3^{-1} 5^{-1}} \right)
\]
\[
= \frac{17}{2} \sum_{u=1}^{\infty} \left( \frac{1}{15} \right)^u \sum_{v=1}^{\infty} \left( \frac{2}{5} \right)^v \sum_{w=1}^{\infty} \left( \frac{2}{3} \right)^w
\]
\[
= \frac{17}{2} \frac{1/15}{1-1/15} \frac{2/5}{1-2/5} \frac{2/3}{1-2/3}
\]
\[
= \frac{17}{21}.
\]

B5 The answer is 4.

Assume \( n \geq 3 \) for the moment. We write the permutations \( \pi \) counted by \( P_n \) as sequences \( \pi(1), \pi(2), \ldots, \pi(n) \). Let \( U_n \) be the number of permutations counted by \( P_n \) that end with \( n-1, n \); let \( V_n \) be the number ending in \( n, n-1 \); let \( W_n \) be the number starting with \( n-1 \) and ending in \( n-2, n \); let \( T_n \) be the number ending in \( n-2, n \) but not starting with \( n-1 \); and let \( S_n \) be the number which has \( n-1, n \) consecutively in that order, but not at the beginning or end. It is clear that every permutation \( \pi \) counted by \( P_n \) either lies in exactly one of the sets counted by \( U_n, V_n, W_n, T_n, S_n \), or is the reverse of such a permutation. Therefore
\[
P_n = 2(U_n + V_n + W_n + T_n + S_n).
\]

By examining how each of the elements in the sets counted by \( U_{n+1}, V_{n+1}, W_{n+1}, T_{n+1}, S_{n+1} \) can be obtained from a (unique) element in one of the sets counted by \( U_n, V_n, W_n, T_n, S_n \) by suitably inserting the element \( n+1 \), we obtain the recurrence relations
\[
U_{n+1} = U_n + W_n + T_n,
\]
\[
V_{n+1} = U_n,
\]
\[
W_{n+1} = W_n,
\]
\[
T_{n+1} = V_n,
\]
\[
S_{n+1} = S_n + V_n.
\]

Also, it is clear that \( W_n = 1 \) for all \( n \).

So far we have assumed \( n \geq 3 \), but it is straightforward to extrapolate the sequences \( P_n, U_n, V_n, W_n, T_n, S_n \) back to \( n = 2 \) to preserve the preceding identities. Hence for all \( n \geq 2 \),
\[
P_{n+5} = 2(U_{n+5} + V_{n+5} + W_{n+5} + T_{n+5} + S_{n+5})
\]
\[
= 2((U_{n+4} + W_{n+4} + T_{n+4}) + U_{n+4}
\]
\[
\quad + W_{n+4} + V_{n+4} + (S_{n+4} + V_{n+4}))
\]
\[
= P_{n+4} + 2(U_{n+4} + W_{n+4} + V_{n+4})
\]
\[
= P_{n+4} + 2((U_{n+3} + W_{n+3} + T_{n+3}) + W_{n+3} + U_{n+3})
\]
\[
= P_{n+4} + P_{n+3} + 2(U_{n+3} - V_{n+3} + W_{n+3} - S_{n+3})
\]
\[
= P_{n+4} + P_{n+3} + 2((U_{n+2} + W_{n+2} + T_{n+2}) - U_{n+2}
\]
\[
\quad + W_{n+2} - (S_{n+2} - V_{n+2}))
\]
\[
= P_{n+4} + P_{n+3} + 2(2W_{n+2} + T_{n+2} - S_{n+2} - V_{n+2})
\]
\[
= P_{n+4} + P_{n+3} + 2(2W_{n+1} + V_{n+1}
\]
\[
\quad - (S_{n+1} + V_{n+1}) - U_{n+1})
\]
\[
= P_{n+4} + P_{n+3} + 2(2W_n + U_n - (S_n + V_n) - U_n
\]
\[
\quad - (U_n + W_n + T_n))
\]
\[
= P_{n+4} + P_{n+3} - P_n + 4,
\]
as desired.

**Remark:** There are many possible variants of the above solution obtained by dividing the permutations up according to different features. For example, Karl Mahlburg suggests writing
\[
P_n = 2P'_n, \qquad P'_n = Q'_n + R'_n
\]
where \( P'_n \) counts those permutations counted by \( P_n \) for which 1 occurs before 2, and \( Q'_n \) counts those permutations counted by \( P'_n \) for which \( \pi(1) = 1 \). One then has the recursion
\[
Q'_n = Q'_{n-1} + Q'_{n-3} + 1
\]