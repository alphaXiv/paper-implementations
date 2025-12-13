<|ref|>image<|/ref|><|det|>[[125, 70, 861, 640]]<|/det|>
<|ref|>image_caption<|/ref|><|det|>[[63, 644, 920, 671]]<|/det|>
<center>Fig. 4. The dispersion contours with stepsizes \(\Delta t = 0.01\) , \(\Delta = 0.1\) for Maxwell's equations (46) from (a) exact dispersion; (b) boxscheme; (c) symplectic method and (d) Yee's method. The constant contour values are \(\omega \in [2,4,6,\ldots ,24]\) . </center>  

<|ref|>equation<|/ref|><|det|>[[108, 684, 922, 720]]<|/det|>
\[\phi = \tan^{-1}\left(\frac{(v_g)_y}{(v_g)_x}\right),\quad |v_g| = \sqrt{(v_g)_x^2 + (v_g)_y^2}. \quad (48)\]  

<|ref|>text<|/ref|><|det|>[[64, 724, 925, 755]]<|/det|>
Substituting into (48) the vectors \(\kappa\) and \(v_{g}\) in polar coordinates (44), and let \(a = |\kappa |\Delta\) , this yields the propagation angle \(\phi\) and the propagation speed \(|v_{g}|\) in terms of \(a\) and \(\theta\) .  

<|ref|>text<|/ref|><|det|>[[87, 754, 410, 769]]<|/det|>
For example, \(\phi\) for the boxscheme is given by  

<|ref|>equation<|/ref|><|det|>[[108, 772, 416, 811]]<|/det|>
\[\phi = \tan^{-1}\left(\frac{\sin\left(\frac{1}{2}\sin(\theta)a\right)\cos^{3}\left(\frac{1}{2}\cos(\theta)a\right)}{\cos^{3}\left(\frac{1}{2}\sin(\theta)a\right)\sin\left(\frac{1}{2}\cos(\theta)a\right)}\right).\]  

<|ref|>text<|/ref|><|det|>[[64, 816, 591, 833]]<|/det|>
Taking the Taylor expansion of this expression with respect to \(a = 0\) yields,  

<|ref|>equation<|/ref|><|det|>[[108, 836, 922, 867]]<|/det|>
\[\phi \approx \theta -\frac{1}{12}\sin (4\theta)a^2 +O(a^3). \quad (49)\]  

<|ref|>text<|/ref|><|det|>[[64, 871, 433, 887]]<|/det|>
Similarly, the Taylor expansion of \(|v_{g}|\) at \(a = 0\) yields,  

<|ref|>equation<|/ref|><|det|>[[108, 890, 922, 924]]<|/det|>
\[|v_g|\approx 1 + \left(\frac{1}{16}\cos (4\theta) - \frac{r^2}{4} +\frac{3}{16}\right)a^2 +O(a^4), \quad (50)\]