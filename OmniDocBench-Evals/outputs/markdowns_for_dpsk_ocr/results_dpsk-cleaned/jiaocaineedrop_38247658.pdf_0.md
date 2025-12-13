
# 2023学年《线性代数B》期末考试试卷  

# 数学  

本试卷共4页，22题，全卷满分150分，考试用时120分钟。  

## 注意事项：  

1. 答卷前，考生务必将自己的姓名、准考证号填写在答题卡上。  

2. 回答选择题时，用铅笔把答题卡上对应题目的答案标号涂黑，写在本试卷上无效。  

3. 考试结束后，将本试卷和答题卡一并交回。  

4. 本试卷由kmath.cn自动生成。  

<table><tr><td>得分</td><td></td></tr><tr><td>阅卷人</td><td></td></tr></table>  

## 一、单选题：本题共10小题，每小题5分，共40分．在每小题给出的四个选项中，只有一项是符合题目要求的.  

1. 设 \(f(x) = \left| \begin{array}{lll}2x & x & 1 & 0 \\ 1 & x & 2 & 3 \\ 2 & 3 & x & 2 \\ 1 & 1 & 2 & 2x \end{array} \right|\) 中，则 \(x^{3}\) 的系数是  

A.-2 B.2 C.4 D.-4  

[答案]：A[解析]：解析： \(x\) 的关联项是 \(a_{12}a_{21}a_{33}a_{44}\) ，前面需要有一个 \((- 1)^{4}\) 他的逆序数是 \(t(2143) = 1\) 所以，- 2  

2. 在下列5阶行列式中，符号为正的项是  

A. \(a_{13}a_{24}a_{32}a_{41}a_{55}\) B. \(a_{15}a_{31}a_{22}a_{44}a_{53}\) C. \(a_{23}a_{32}a_{41}a_{15}a_{54}\) D. \(a_{31}a_{25}a_{43}a_{14}a_{52}\)  

[答案]：B[解析]：  

3. 已知矩阵 \(A = \left( \begin{array}{lll}1 & 2 & 1\\ -1 & 0 & 1\\ 0 & 1 & 0 \end{array} \right),A^{*}\) 为 \(A\) 的伴随矩阵，则 \(\left|A^{*}\right| =\)  

A. \(- \frac{1}{2}\) B. \(\frac{1}{4}\) C.2 
D.4  

[答案]：D[解析]：先记住一个结论 \(\left|A^{*}\right| = \left|A\right|^{n - 1}\) 其中 \(n\) 为行列式的阶数．所以， \(\left|A^{*}\right| = \left|A\right|^{2} = 4\)  

4. 设 \(A = \left( \begin{array}{ll}2 & 3\\ 3 & 0\\ 1 & 2 \end{array} \right),B = \left( \begin{array}{ll}2 & 1 & 3\\ -3 & 0 & -1 \end{array} \right)\) ，则 \(\left|AB\right| =\)  

A.0 
B.4 
C.6 
D.8  

[答案]：A[解析]：本题，很多同学容易想到 \(\left|AB\right| = \left|A\right|\cdot \left|B\right|\) 但是，他的前提是 \(A,B\) 都是方阵，所以本题不能直接使用这个公式，计算可得  

\[A B = \left( \begin{array}{c c c}{-5 2 3\] \[6 3 9\] \[-4 1 1} \end{array} \right)\qquad \left|A B\right| = \left| \begin{array}{c c c}{-5 2 3\] \[6 3 9\] \[-4 1 1} \end{array} \right|\]  

然后利用初等变换可以得到结果是0  

5. 已知 \(A,B\) 都是 \(n\) 阶矩阵，且 \(AB = 0\) ，则必有  

A. \(A = 0\) 或 \(B = 0\) B. \(\left|A\right| = \left|B\right| = 0\) C. \(A = B = 0\) D. \(\left|A\right| = 0\) 或 \(\left|B\right| = 0\)  

[答案]：D[解析]：本题考查的基本概念，例如 \(A = \left( \begin{array}{ll}1 & 0\\ 0 & 0 \end{array} \right)\) \(B = \left( \begin{array}{ll}0 & 0\\ 0 & 1 \end{array} \right)\) \(AB = \left( \begin{array}{ll}0 & 0\\ 0 & 0 \end{array} \right)\) 但是 \(A,B\) 都不是零.  

由 \(AB = 0\) 同时取行列式 \(\left|AB\right| = \left|0\right| = 0\) 所以，选D \(\left|A\right|\cdot \left|B\right| = 0\)  

6. 可量组 \(a_{1} = (1,1,1,1)^{T},a_{2} = (1,2,3,4)^{T},a_{3} = (0,1,2,3)^{T}\) 的秩为  

A.1 
B.2 
C.3 
D.4  

[答案]：B[解析]：根据三秩相等定理：列向量组的秩等于列向量组所构成的秩，等于矩阵行向量组的秩. \(R(a_{1},a_{2},a_{3}) = R(A)\) 而  

\[A = \left( \begin{array}{lll}1 1 0\] \[1 2 1\] \[1 3 2\] \[1 4 3} \end{array} \right)\]  

对A进行初等变换求秩  

\[A = \left( \begin{array}{lll}1 1 0\] \[1 2 1\] \[1 3 2\] \[1 4 3} \end{array} \right)\leftrightarrow \left( \begin{array}{lll}1 1 0\] \[0 1 1\] \[0 0 0\] \[0 0 0} \end{array} \right)\]  

可见秩为2  

7. 设 \(A\) 是3阶矩阵，且 \(\left|A\right| = -2\) 则 \(\left|\left(\frac{1}{12} A\right)^{-1} + (3A)^{*}\right| =\)  

A.-108 
B.108 
C.54 
D.-54  

[答案]：B[解析]：对于矩阵有一个性质，如果A可逆，那么 \((\lambda A)^{- 1} = \frac{1}{\lambda}\cdot A^{- 1}\) 所以， \(\left(\frac{1}{12} A\right)^{- 1} = 12A^{- 1}\) 而对于伴随矩阵的性质有 \(A^{- 1} = \frac{1}{|A|}\cdot A^{*}\)