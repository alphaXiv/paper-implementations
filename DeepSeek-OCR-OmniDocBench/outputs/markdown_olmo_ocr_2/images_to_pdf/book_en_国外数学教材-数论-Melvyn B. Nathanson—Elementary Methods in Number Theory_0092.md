Theorem 2.19 Let m be an integer that is the product of two prime numbers. The prime divisors of m are the roots of the quadratic equation

\[
x^2 - (m + 1 - \varphi(m))x + m = 0,
\]

and so \( \varphi(m) \) determines the prime factors of m.

Proof. If \( m = pq \), then

\[
\varphi(m) = (p-1)(q-1) = pq - p - q + 1 = m - p - \frac{m}{p} + 1,
\]

and so

\[
p - (m + 1 - \varphi(m)) + \frac{m}{p} = 0.
\]

Equivalently, p and q are the solutions of the quadratic equation

\[
x^2 - (m + 1 - \varphi(m))x + m = 0.
\]

This completes the proof. \( \Box \)

For example, if \( m = 221 \) and \( \varphi(m) = 192 \), then the quadratic equation

\[
x^2 - 30x + 221 = 0
\]

has solutions \( x = 13 \) and \( x = 17 \), and \( 221 = 13 \cdot 17 \).

This method, known as the RSA cryptosystem, is called a public key cryptosystem, since the encryption key is made available to everyone, and the encrypted message can be transmitted through public channels. Only the possessor of the prime factors of m can decrypt the message. RSA is simple, but useful, and is the basis of many commercially valuable cryptosystems.

Exercises

1. Consider the secret key cryptosystem constructed from the prime \( p = 947 \) and the encoding key \( e = 167 \). Encipher the plaintext \( P = 2 \). Find a decrypting key and decipher the ciphertext \( C = 3 \).

2. Consider the primes \( p = 53 \) and \( q = 61 \). Let \( m = pq \). Prove that \( e = 7 \) is relatively prime to \( \varphi(m) \). Find a positive integer d such that \( ed \equiv 1 \pmod{\varphi(m)} \).

3. The integer 6059 is the product of two distinct primes, and \( \varphi(6059) = 5904 \). Use Theorem 2.19 to compute the prime divisors of 6059.

4. The probability that an integer chosen at random between 1 and n is relatively prime to n is \( \varphi(n)/n \). Let \( n = pq \), where p and q are distinct primes greater than x. Prove that the probability that a randomly chosen positive integer up to x is relatively prime to n is greater than \( (1 - 1/x)^2 \). If \( x = 200 \), this probability is greater than 0.99.