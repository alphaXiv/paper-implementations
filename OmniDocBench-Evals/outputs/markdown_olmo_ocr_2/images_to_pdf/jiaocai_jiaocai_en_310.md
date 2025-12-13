Zhejiang University
Theory of Computation, Fall-Winter 2015
Final Exam

1. (20%) Determine whether the following statements are true or false. If it is true write a \( \circ \) otherwise a \( \times \) in the bracket before the statement.

(a) ( ) Let \( L \) be any regular language. Then the number of equivalence classes respect to language \( L \) (i.e. \( x \approx_L y \), if for all \( z \in \Sigma^* \), \( xz \in L \) iff \( yz \in L \) ) is finite.

(b) ( ) The language \( \{a^n b^n w | w \in \{a,b\}^*, n \in \mathbb{N}, \text{ and } |w| = 2n\} \) is context-free.

(c) ( ) The language \( \{“M” “w” | M \text{ accepts } w \text{ in less than 2016 steps }\} \) is not recursive.

(d) ( ) The language \( \{“M” | M \text{ is a TM and } L(M) \text{ is Context-free but } L(M) \text{ is not regular}\} \) is not recursive.

(e) ( ) The language \( \{“M_1” “M_2” | M_1 \text{ and } M_2 \text{ are TMs, and } M_1 \text{ halts on } e \text{ but } M_2 \text{ doesn’t halt on } e\} \) is recursively enumerable, but not recursive.

(f) ( ) The set of all primitive recursive functions is a proper subset of the set of all \( \mu \)-recursive functions.

(g) ( ) Let \( A \) and \( B \) be two disjoint recursively enumerable languages. If \( \overline{A \cup B} \) is also be recursively enumerable, then it is possible that neither \( A \) nor \( B \) is decidable.

(h) ( ) Let \( H_{10} = \{“M” | M \text{ is a TM and } 10 \in L(M)\} \) and \( \tau_1 \) and \( \tau_2 \) are recursive functions. If \( H_{10} \leq_{\tau_1} L \) and \( \overline{H_{10}} \leq_{\tau_2} L \), then \( L \) is recursive enumerable but not recursive.

(i) ( ) Suppose \( \mathbf{SAT} \leq_P L \) and \( L \in \mathbb{P} \). Then \( \mathbb{P} = \mathbb{NP} \).

(j) ( ) Let \( H = \{“M” “w” | \text{TM } M \text{ halts on input string } w\} \), then \( H \) is \( \mathbb{NP} \)-complete.