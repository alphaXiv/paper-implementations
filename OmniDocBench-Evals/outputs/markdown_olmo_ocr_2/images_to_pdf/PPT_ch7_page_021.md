Parallel processing of products on the computer is facilitated by a variant of (3) for computing \( C = AB \), which is used by standard algorithms (such as in Lapack). In this method, \( A \) is used as given, \( B \) is taken in terms of its column vectors, and the product is computed columnwise; thus,

(5) \( AB = A[b_1\ b_2\ ... \ b_p] = [Ab_1\ Ab_2\ ... \ Ab_p] \).

Columns of \( B \) are then assigned to different processors (individually or several to each processor), which simultaneously compute the columns of the product matrix \( Ab_1, Ab_2, \) etc.