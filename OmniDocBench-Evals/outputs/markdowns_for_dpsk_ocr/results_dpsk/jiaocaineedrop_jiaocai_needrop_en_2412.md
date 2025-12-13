<|ref|>text<|/ref|><|det|>[[87, 85, 460, 119]]<|/det|>
设 \(u(x) = \ln \left(x + 1\right) - x\) ,则 \(u^{\prime}(x) = \frac{1}{x + 1} - 1 = \frac{- x}{x + 1}\) ,  

<|ref|>text<|/ref|><|det|>[[87, 125, 511, 144]]<|/det|>
所以当 \(- 1< x< 0\) 时, \(u^{\prime}(x) > 0\) ;当 \(x > 0\) 时, \(u^{\prime}(x)< 0\)  

<|ref|>text<|/ref|><|det|>[[87, 154, 508, 172]]<|/det|>
所以 \(u(x)\) 在 \((- 1,0)\) 上为增函数,在 \((0, + \infty)\) 上为减函数,  

<|ref|>text<|/ref|><|det|>[[87, 183, 430, 201]]<|/det|>
故 \(u(x)_{\mathrm{max}} = u(0) = 0\) ,所以 \(\ln \left(x + 1\right)\leq x\) 成立.  

<|ref|>text<|/ref|><|det|>[[87, 211, 639, 245]]<|/det|>
由上还不等式可得,当 \(t > 1\) 时, \(\ln \left(1 + \frac{1}{t}\right)\leq \frac{1}{t} < \frac{2}{t + 1}\) ,故 \(S^{\prime}(t)< 0\) 恒成立,  

<|ref|>text<|/ref|><|det|>[[87, 255, 456, 273]]<|/det|>
故 \(S(t)\) 在 \((1, + \infty)\) 上为减函数,则 \(S(t)< S(1) = 0\)  

<|ref|>text<|/ref|><|det|>[[87, 285, 485, 303]]<|/det|>
所以 \((t - 1)\ln (t + 1) - t\ln t< 0\) 成立,即 \(x_{1} + x_{2}< \mathrm{e}\) 成立.  

<|ref|>text<|/ref|><|det|>[[87, 315, 340, 333]]<|/det|>
综上所述, \(\ln \left(a + b\right)< \ln \left(ab\right) + 1\)  

<|ref|>sub_title<|/ref|><|det|>[[122, 342, 320, 359]]<|/det|>
## 核心考点五:极最值问题  

<|ref|>sub_title<|/ref|><|det|>[[128, 367, 225, 383]]<|/det|>
## 【规律方法】  

<|ref|>text<|/ref|><|det|>[[86, 391, 904, 481]]<|/det|>
利用导数求函数的极最值问题.解题方法是利用导函数与单调性关系确定单调区间,从而求得极最值.只是对含有参数的极最值问题,需要对导函数进行二次讨论,对导函数或其中部分函数再一次求导,确定单调性,零点的存在性及唯一性等,由于零点的存在性与参数有关,因此对函数的极最值又需引入新函数,对新函数再用导数进行求值、证明等操作.  

<|ref|>sub_title<|/ref|><|det|>[[128, 489, 225, 505]]<|/det|>
## 【典型例题】  

<|ref|>text<|/ref|><|det|>[[87, 515, 848, 545]]<|/det|>
例14.(2023春·江西鹰潭·高三贵溪市实验中学校考阶段练习)已知函数 \(f\left(x\right) = \frac{1}{3} x^{3} - a x + a,a\in \mathbf{R}\)  

<|ref|>text<|/ref|><|det|>[[87, 555, 420, 572]]<|/det|>
(1)当 \(a = -1\) 时,求 \(f(x)\) 在 \([-2,2]\) 上的最值;  

<|ref|>text<|/ref|><|det|>[[87, 584, 317, 600]]<|/det|>
(2)讨论 \(f(x)\) 的极值点的个数.  

<|ref|>text<|/ref|><|det|>[[90, 612, 523, 644]]<|/det|>
【解析】(1)当 \(a = -1\) 时, \(f(x) = \frac{1}{3} x^{3} + x - 1\) , \(x\in [-2,2]\)  

<|ref|>text<|/ref|><|det|>[[90, 654, 458, 671]]<|/det|>
\(f^{\prime}(x) = x^{2} + 1 > 0\) ,故 \(f(x)\) 在 \([-2,2]\) 上单调递增,  

<|ref|>equation<|/ref|><|det|>[[88, 681, 470, 713]]<|/det|>
\[\therefore f\left(x\right)_{\mathrm{min}} = f\left(-2\right) = -\frac{17}{3},\quad \therefore f\left(x\right)_{\mathrm{max}} = f\left(2\right) = \frac{11}{3}.\]  

<|ref|>text<|/ref|><|det|>[[97, 722, 242, 739]]<|/det|>
(2) \(f^{\prime}(x) = x^{2} - a\)  

<|ref|>text<|/ref|><|det|>[[87, 747, 618, 765]]<|/det|>
\(①\) 当 \(a\leqslant 0\) 时, \(f^{\prime}(x)\geqslant 0\) 恒成立,此时 \(f(x)\) 在 \(\mathbf{R}\) 上单调递增,不存在极值点.  

<|ref|>text<|/ref|><|det|>[[87, 773, 562, 791]]<|/det|>
\(②\) 当 \(a > 0\) 时,令 \(f^{\prime}(x) > 0\) ,即 \(x^{2} - a > 0\) ,解得: \(x< - \sqrt{a}\) 或 \(x > \sqrt{a}\)  

<|ref|>text<|/ref|><|det|>[[88, 800, 433, 818]]<|/det|>
令 \(f^{\prime}(x)< 0\) ,即 \(x^{2} - a< 0\) ,解得 \(x\in (- \sqrt{a},\sqrt{a})\)  

<|ref|>text<|/ref|><|det|>[[87, 828, 580, 846]]<|/det|>
故此时 \(f(x)\) 在 \((- \infty , - \sqrt{a})\) 递增,在 \((- \sqrt{a},\sqrt{a})\) 递减,在 \((\sqrt{a}, + \infty)\) 递增,  

<|ref|>text<|/ref|><|det|>[[87, 856, 741, 874]]<|/det|>
所以 \(f(x)\) 在 \(x = - \sqrt{a}\) 时取得极大值,在 \(x = \sqrt{a}\) 时取得极小值,故此时极值点个数为2,  

<|ref|>text<|/ref|><|det|>[[87, 883, 333, 899]]<|/det|>
综上所述: \(a\leqslant 0\) 时, \(f(x)\) 无极值点,