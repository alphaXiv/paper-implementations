8. Let \( p \) be an odd prime. Prove that the Legendre symbol is a homomorphism from the multiplicative group \((\mathbb{Z}/p\mathbb{Z})^\times\) into \(\{\pm 1\}\). What is the kernel of this homomorphism?

9. For every odd prime \( p \), define the *Mersenne number*

\[
M_p = 2^p - 1.
\]

A prime number of the form \( M_p \) is called a *Mersenne prime* (see Exercise 5 in Section 1.5).

Let \( q \) be a prime divisor of \( M_p \).

(a) Prove that 2 has order \( p \) modulo \( q \), and so \( p \) divides \( q - 1 \).
    *Hint*: Fermat's theorem.

(b) Prove that \( p \) divides \( (q-1)/2 \), and so

\[
q \equiv 1 \pmod{2p}
\]

and

\[
2^{(q-1)/2} \equiv 1 \pmod{q}.
\]

*Hint*: Both \( p \) and \( q \) are odd.

(c) Prove that \( \left( \frac{2}{q} \right) = 1 \), and so \( q \equiv \pm 1 \pmod{8} \).

10. For every positive integer \( n \), define the *Fermat number*

\[
F_n = 2^{2^n} + 1.
\]

A prime number of the form \( F_n \) is called a *Fermat prime* (see Exercise 7 in Section 1.5).

Let \( n \geq 2 \), and let \( q \) be a prime divisor of \( F_n \).

(a) Prove that 2 has order \( 2^{n+1} \) modulo \( q \).
    *Hint*: Exercise 8 in Section 2.5.

(b) Prove that

\[
q \equiv 1 \pmod{2^{n+1}}.
\]

(c) Prove that there exists an integer \( a \) such that

\[
a^{2^{n+1}} \equiv -1 \pmod{q}.
\]

*Hint*: Observe that \( \left( \frac{2}{q} \right) = 1 \), and so \( 2 \equiv a^2 \pmod{q} \).

(d) Prove that

\[
q \equiv 1 \pmod{2^{n+2}}.
\]