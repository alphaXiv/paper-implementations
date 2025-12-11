1
Divisibility and Primes

1.1 Division Algorithm

Divisibility is a fundamental concept in number theory. Let \( a \) and \( d \) be integers. We say that \( d \) is a divisor of \( a \), and that \( a \) is a multiple of \( d \), if there exists an integer \( q \) such that
\[
a = dq.
\]
If \( d \) divides \( a \), we write
\[
d | a.
\]
For example, 1001 is divisible by 7 and 13. Divisibility is transitive: If \( a \) divides \( b \) and \( b \) divides \( c \), then \( a \) divides \( c \) (Exercise 14).

The minimum principle states that every nonempty set of integers bounded below contains a smallest element. For example, a nonempty set of nonnegative integers must contain a smallest element. We can see the necessity of the condition that the nonempty set be bounded below by considering the example of the set \( \mathbf{Z} \) of all integers, positive, negative, and zero.

The minimum principle is all we need to prove the following important result.

**Theorem 1.1 (Division algorithm)** *Let \( a \) and \( d \) be integers with \( d \geq 1 \). There exist unique integers \( q \) and \( r \) such that*
\[
a = dq + r \tag{1.1}
\]
*and*
\[
0 \leq r \leq d - 1. \tag{1.2}
\]