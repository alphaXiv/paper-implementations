<|ref|>text<|/ref|><|det|>[[147, 93, 614, 134]]<|/det|>
例4:已知数列 \(\left\{a_{n}\right\}\) 满足 \(a_{1} = 2,a_{n + 1} = 2{\left(1 + \frac{1}{n}\right)}^{2}a_{n},n\in N_{+}\)  

<|ref|>text<|/ref|><|det|>[[155, 147, 657, 187]]<|/det|>
(1)求证:数列 \(\left\{\frac{a_{n}}{n^{2}}\right\}\) 是等比数列,并求出数列 \(\left\{a_{n}\right\}\) 的通项公式  

<|ref|>text<|/ref|><|det|>[[155, 203, 515, 242]]<|/det|>
(2)设 \(c_{n} = \frac{n}{a_{n}}\) ,求证: \(c_{1} + c_{2} + \dots +c_{n}< \frac{17}{24}\)  

<|ref|>text<|/ref|><|det|>[[147, 258, 500, 300]]<|/det|>
解:(1) \(a_{n + 1} = 2{\left(1 + \frac{1}{n}\right)}^{2}a_{n} = 2\cdot {\frac{\left(n + 1\right)^{2}}{n^{2}}}a_{n}\)  

<|ref|>text<|/ref|><|det|>[[149, 315, 558, 353]]<|/det|>
\(\therefore \frac{a_{n + 1}}{\left(n + 1\right)^{2}} = 2\cdot \frac{a_{n}}{n^{2}}\quad \therefore \left\{\frac{a_{n}}{n^{2}}\right\}\) 是公比为2的等比数列  

<|ref|>equation<|/ref|><|det|>[[149, 370, 331, 410]]<|/det|>
\[\therefore \frac{a_{n}}{n^{2}} = \left(\frac{a_{1}}{1^{2}}\right)\cdot 2^{n - 1} = 2^{n}\]  

<|ref|>equation<|/ref|><|det|>[[150, 425, 258, 446]]<|/det|>
\[\therefore a_{n} = n^{2}\cdot 2^{n}\]  

<|ref|>text<|/ref|><|det|>[[147, 460, 850, 567]]<|/det|>
(2)思路: \(c_{n} = \frac{n}{a_{n}} = \frac{1}{n\cdot2^{n}}\) ,无法直接求和,所以考虑放缩成为可求和的通项公式(不等号: \(< >\) ),若要放缩为裂项相消的形式,那么需要构造出"顺序同构"的特点。观察分母中有 \(n\) ,故分子分母通乘以 \((n - 1)\) ,再进行放缩调整为裂项相消形式。  

<|ref|>text<|/ref|><|det|>[[147, 580, 420, 620]]<|/det|>
解: \(c_{n} = \frac{n}{a_{n}} = \frac{1}{n\cdot2^{n}} = \frac{n - 1}{n(n - 1)2^{n}}\)  

<|ref|>equation<|/ref|><|det|>[[147, 636, 533, 678]]<|/det|>
\[\frac{1}{(n - 1)2^{n - 1}} -\frac{1}{n\cdot2^{n}} = \frac{2n - (n - 1)}{n(n - 1)2^{n}} = \frac{n + 1}{n(n - 1)2^{n}}\]  

<|ref|>text<|/ref|><|det|>[[147, 693, 647, 733]]<|/det|>
所以 \(c_{n} = \frac{n - 1}{n(n - 1)2^{n}} < \frac{n + 1}{n(n - 1)2^{n}} = \frac{1}{(n - 1)2^{n - 1}} -\frac{1}{n\cdot2^{n}} (n\geq 2)\)  

<|ref|>equation<|/ref|><|det|>[[147, 747, 832, 870]]<|/det|>
\[c_{1} + c_{2} + \dots +c_{n}< c_{1} + c_{2} + c_{3} + \left(\frac{1}{3\cdot2^{3}} -\frac{1}{4\cdot2^{4}} +\frac{1}{4\cdot2^{4}} -\frac{1}{5\cdot2^{5}} +\dots +\frac{1}{(n - 1)2^{n - 1}} -\frac{1}{n\cdot2^{n}}\right)\] \[\qquad = \frac{1}{2} +\frac{1}{8} +\frac{1}{24} +\frac{1}{24} -\frac{1}{n\cdot2^{n}} = \frac{17}{24} -\frac{1}{n\cdot2^{n}} < \frac{17}{24} (n > 3)\] \[\because c_{n} > 0\quad \therefore c_{1}< c_{1} + c_{2}< c_{1} + c_{2} + c_{3} = \frac{16}{24} < \frac{17}{24}\]  

<|ref|>text<|/ref|><|det|>[[147, 875, 850, 894]]<|/det|>
小炼有话说:(1)本题先确定放缩的类型,向裂项相消放缩,从而按"依序同构"的目标进