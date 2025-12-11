<|ref|>text<|/ref|><|det|>[[70, 65, 919, 111]]<|/det|>
10. Proof. We omit (a) since is standard. For (b), if \(u\) attains an interior maximum, then the conclusion follows from strong maximum principle.  

<|ref|>text<|/ref|><|det|>[[104, 115, 917, 162]]<|/det|>
If not, then for some \(x^{0}\in \partial U,u(x^{0}) > u(x)\forall x\in U\) . Then Hopf's lemma implies \(\begin{array}{r}{\frac{\partial u}{\partial\nu} (x^{0}) > 0} \end{array}\) which is a contradiction.  

<|ref|>text<|/ref|><|det|>[[105, 173, 917, 219]]<|/det|>
Remark 2. A generalization of this problem to mixed boundary conditions is recorded in Gilbarg-Trudinger, Elliptic PDEs of second order, Problem 3.1.  

<|ref|>text<|/ref|><|det|>[[70, 234, 226, 252]]<|/det|>
11. Proof. Define  

<|ref|>equation<|/ref|><|det|>[[272, 252, 748, 294]]<|/det|>
\[B[u,v] = \int_{U}\sum_{i,j}a^{i j}u_{x_{i}}v_{x_{j}}d x\mathrm{~for~}u\in H^{1}(U),v\in H_{0}^{1}(U).\]  

<|ref|>text<|/ref|><|det|>[[104, 298, 672, 320]]<|/det|>
By Exercise 5.17, \(\phi (u)\in H^{1}(U)\) . Then, for all \(v\in C_{c}^{\infty}(U)\) , \(v\geq 0\)  

<|ref|>equation<|/ref|><|det|>[[189, 330, 836, 499]]<|/det|>
\[B[\phi (u),v] = \int_{U}\sum_{i,j}a^{i j}(\phi (u))_{x_{i}}v_{x_{j}}d x\] \[\qquad = \int_{U}\sum_{i,j}a^{i j}\phi^{\prime}(u)u_{x_{i}}v_{x_{j}}d x,(\phi^{\prime}(u)\mathrm{~is~bounded~since~}u\mathrm{~is~bounded})\] \[\qquad = \int_{U}\sum_{i,j}a^{i j}u_{x_{i}}(\phi^{\prime}(u)v)_{x_{j}} - \sum_{i,j}a_{i j}\phi^{\prime \prime}(u)u_{x_{i}}u_{x_{j}}v d x\] \[\qquad \leq 0 - \int_{U}\phi^{\prime \prime}(u)v|D u|^{2}d x\leq 0,\mathrm{~by~convexity~of~}\phi .\]  

<|ref|>text<|/ref|><|det|>[[104, 504, 917, 551]]<|/det|>
(We don't know whether the product of two \(H^{1}\) functions is weakly differentiable. This is why we do not take \(v\in H_{0}^{1}\) .) Now we complete the proof with the standard density argument. \(\square\)  

<|ref|>text<|/ref|><|det|>[[70, 566, 917, 614]]<|/det|>
12. Proof. Given \(u\in C^{2}(U)\cap C(\bar{U})\) with \(Lu\leq 0\) in \(U\) and \(u\leq 0\) on \(\partial U\) . Since \(\bar{U}\) is compact and \(v\in C(\bar{U})\) , \(v\geq c > 0\) . So \(w\coloneqq \frac{u}{v}\in C^{2}(U)\cap C(\bar{U})\) . Brutal computation gives us  

<|ref|>equation<|/ref|><|det|>[[135, 622, 794, 735]]<|/det|>
\[-a^{i j}w_{x_{i}x_{j}} = \frac{-a^{i j}u_{x_{i}x_{j}}v + a^{i j}v_{x_{i}x_{j}}u}{v^{2}} +\frac{a^{i j}v_{x_{i}}u_{x_{j}} - a^{i j}u_{x_{i}}v_{x_{j}}}{v^{2}} -a^{i j}\frac{2}{v}v_{x_{j}}\frac{v_{x_{i}}u - v u_{x_{i}}}{v^{2}}\] \[\qquad = \frac{(L u - b^{i}u_{x_{i}} - c u)v + (-L v + b^{i}v_{x_{i}} + c v)u}{v^{2}} +0 + a^{i j}\frac{2}{v}v_{x_{j}}w_{x_{i}},\mathrm{~since~}a^{i j} = a^{i j}.\] \[\qquad = \frac{L u}{v} -\frac{u L v}{v^{2}} -b^{i}w_{x_{i}} + a^{i j}\frac{2}{v}v_{x_{j}}w_{x_{i}}\]  

<|ref|>text<|/ref|><|det|>[[105, 742, 196, 759]]<|/det|>
Therefore,  

<|ref|>equation<|/ref|><|det|>[[165, 770, 857, 806]]<|/det|>
\[M w\coloneqq -a^{i j}w_{x_{i}x_{j}} + w_{x_{i}}\big[b^{i} - a^{i j}\frac{2}{v} v_{x_{j}}\big] = \frac{L u}{v} -\frac{u L v}{v^{2}}\leq 0\mathrm{~on~}\{x\in \bar{U}:u > 0\} \subseteq U\]  

<|ref|>text<|/ref|><|det|>[[104, 816, 917, 862]]<|/det|>
If \(\{x\in \bar{U}:u > 0\}\) is not empty, Weak maximum principle to the operator \(M\) with bounded coefficients (since \(v\in C^{1}(\bar{U})\) ) will lead a contradiction that  

<|ref|>equation<|/ref|><|det|>[[375, 875, 646, 908]]<|/det|>
\[0< \max_{\{u > 0\}}w = \max_{\partial \{u > 0\}}w = \frac{0}{v} = 0\]  

<|ref|>text<|/ref|><|det|>[[105, 918, 264, 936]]<|/det|>
Hence \(u\leq 0\) in \(U\)