Proof. This follows immediately from Theorem 3.3, since \( |(\mathbb{Z}/p\mathbb{Z})^\times| = p-1 \). □

The following table lists the primitive roots for the first six primes.

<table>
  <tr>
    <th>p</th>
    <th>\( \varphi(p-1) \)</th>
    <th>primitive roots</th>
  </tr>
  <tr>
    <td>2</td>
    <td>1</td>
    <td>1</td>
  </tr>
  <tr>
    <td>3</td>
    <td>1</td>
    <td>2</td>
  </tr>
  <tr>
    <td>5</td>
    <td>2</td>
    <td>2,3</td>
  </tr>
  <tr>
    <td>7</td>
    <td>2</td>
    <td>3,5</td>
  </tr>
  <tr>
    <td>11</td>
    <td>4</td>
    <td>2,6,7,8</td>
  </tr>
  <tr>
    <td>13</td>
    <td>4</td>
    <td>2,6,7,11</td>
  </tr>
</table>

Let \( p \) be a prime, and let \( g \) be a primitive root modulo \( p \). If \( a \) is an integer not divisible by \( p \), then there exists a unique integer \( k \) such that
\[
a \equiv g^k \pmod{p}
\]
and
\[
k \in \{0, 1, \ldots, p-2\}.
\]
This integer \( k \) is called the *index* of \( a \) with respect to the primitive root \( g \), and is denoted by
\[
k = \mathrm{ind}_g(a).
\]
If \( k_1 \) and \( k_2 \) are any integers such that \( k_1 \leq k_2 \) and
\[
a \equiv g^{k_1} \equiv g^{k_2} \pmod{p},
\]
then
\[
g^{k_2 - k_1} \equiv 1 \pmod{p},
\]
and so
\[
k_1 \equiv k_2 \pmod{p-1}.
\]
If \( a \equiv g^k \pmod{p} \) and \( b \equiv g^\ell \pmod{p} \), then \( ab \equiv g^k g^\ell = g^{k+\ell} \pmod{p} \), and so
\[
\mathrm{ind}_g(ab) \equiv k + \ell \equiv \mathrm{ind}_g(a) + \mathrm{ind}_g(b) \pmod{p-1}.
\]
The index map \( \mathrm{ind}_g \) is also called the *discrete logarithm* to the base \( g \) modulo \( p \).

For example, 2 is a primitive root modulo 13. Here is a table of \( \mathrm{ind}_2(a) \) for \( a = 1, \ldots, 12 \):

<table>
  <tr>
    <th>a</th>
    <th>\( \mathrm{ind}_2(a) \)</th>
    <th>a</th>
    <th>\( \mathrm{ind}_2(a) \)</th>
  </tr>
  <tr>
    <td>1</td>
    <td>0</td>
    <td>7</td>
    <td>11</td>
  </tr>
  <tr>
    <td>2</td>
    <td>1</td>
    <td>8</td>
    <td>3</td>
  </tr>
  <tr>
    <td>3</td>
    <td>4</td>
    <td>9</td>
    <td>8</td>
  </tr>
  <tr>
    <td>4</td>
    <td>2</td>
    <td>10</td>
    <td>10</td>
  </tr>
  <tr>
    <td>5</td>
    <td>9</td>
    <td>11</td>
    <td>7</td>
  </tr>
  <tr>
    <td>6</td>
    <td>5</td>
    <td>12</td>
    <td>6</td>
  </tr>
</table>