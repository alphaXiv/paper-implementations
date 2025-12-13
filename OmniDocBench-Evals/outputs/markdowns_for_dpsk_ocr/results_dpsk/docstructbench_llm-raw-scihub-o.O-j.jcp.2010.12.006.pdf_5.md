<|ref|>equation<|/ref|><|det|>[[115, 66, 750, 101]]<|/det|>
\[\partial_{t}\mathcal{Z}_{i,j,k}^{l} = \frac{z_{i,j,k}^{l + 1} - z_{i,j,k}^{l}}{\Delta t},\quad \overline{\partial}_{t}\mathcal{Z}_{i,j,k}^{l} = \frac{z_{i,j,k}^{l + 1} - z_{i,j,k}^{l - 1}}{2\Delta t},\quad \partial_{x_{1}}\mathcal{Z}_{i,j,k}^{l} = \frac{z_{i,j,k}^{l} - z_{i,j,k}^{l}}{2\Delta x_{1}},\quad \overline{\partial}_{x_{1}}\mathcal{Z}_{i,j,k}^{l} = \frac{z_{i,j,k}^{l} - z_{i,j,k}^{l - 1}}{2\Delta x_{1}},\]  

<|ref|>text<|/ref|><|det|>[[72, 104, 933, 150]]<|/det|>
where the difference operators \(\partial_{t}\) , \(\overline{\partial}_{t}\) are the discretization of \(\frac{\partial}{\partial t}\) , and the difference operators \(\partial_{x_{1}}\) , \(\overline{\partial}_{x_{1}}\) are the discretization of \(\frac{\partial}{\partial x_{1}}\) . Via these notations and the properties of the difference operators, we present the statement of the local/global conservation laws for the different numerical methods.  

<|ref|>sub_title<|/ref|><|det|>[[73, 163, 392, 178]]<|/det|>
### 3.1. Symplectic method for Maxwell's equations  

<|ref|>text<|/ref|><|det|>[[72, 191, 933, 236]]<|/det|>
The following method is constructed based on the method of lines, i.e. discretizing the Hamiltonian PDEs in space, then applying the symplectic method to the resulting Hamiltonian ODEs (see for example [6,22]). Later, we will show the method is also multisymplectic in the corresponding statement of multisymplecticity.  

<|ref|>text<|/ref|><|det|>[[72, 234, 932, 278]]<|/det|>
For Maxwell's equations in Hamiltonian form, we use the central finite difference in space (which is leapfrog discretization) and implicit midpoint rule (which is symplectic) in time, it is easy to show that the Hamiltonian formulations in (5) and (7) for Maxwell's equations reduce to the same discretized system,  

<|ref|>equation<|/ref|><|det|>[[115, 282, 930, 305]]<|/det|>
\[\partial_{t}\mathcal{Z}_{i,j,k}^{l} + M^{-1}K_{1}\overline{\partial}_{x_{1}}\mathcal{Z}_{i,j,k}^{l + 1} + M^{-1}K_{2}\overline{\partial}_{x_{2}}\mathcal{Z}_{i,j,k}^{l + 1} + M^{-1}K_{3}\overline{\partial}_{x_{3}}\mathcal{Z}_{i,j,k}^{l + 1} = 0, \quad (18)\]  

<|ref|>text<|/ref|><|det|>[[72, 310, 930, 340]]<|/det|>
where indices \(i,j,k\) denote spatial increments and index \(l\) denotes time increment, and matrices \(M,K_{1},\ldots\) as in (10). We refer to this particular discretization as the symplectic method, though it is also multisymplectic.  

<|ref|>text<|/ref|><|det|>[[72, 339, 932, 382]]<|/det|>
The symplectic method (18) is second- order in space and time, and is unconditionally stable. Furthermore, this discrete system preserves two discretized global conservation laws: the first one is the discrete quadratic global conservation law based on (8),  

<|ref|>equation<|/ref|><|det|>[[112, 384, 930, 415]]<|/det|>
\[\frac{1}{2}\partial_{t}\left[\mu \mathbf{H}_{i,j,k}^{l}\cdot \mathbf{H}_{i,j,k}^{l} + \epsilon \mathbf{E}_{i,j,k}^{l}\cdot \mathbf{E}_{i,j,k}^{l}\right] = 0. \quad (19)\]  

<|ref|>text<|/ref|><|det|>[[75, 419, 927, 437]]<|/det|>
The second discretized global conservation law for symplectic method is based on the helicity Hamiltonian functional (6)  

<|ref|>equation<|/ref|><|det|>[[112, 440, 930, 472]]<|/det|>
\[\partial_{t}\left[\frac{1}{2\epsilon}\mathbf{H}_{i,j,k}^{l}\cdot \widehat{\nabla}\times \mathbf{H}_{i,j,k}^{l} + \frac{1}{2\mu}\mathbf{E}_{i,j,k}^{l}\cdot \widehat{\nabla}\times \mathbf{E}_{i,j,k}^{l}\right] = 0, \quad (20)\]  

<|ref|>text<|/ref|><|det|>[[72, 476, 930, 508]]<|/det|>
where \(\widehat{\nabla}\times = R_{1}\partial_{x_{1}} + R_{2}\partial_{x_{2}} + R_{3}\partial_{x_{3}}\) . Furthermore, the scheme (18) is proved to be multisymplectic, since it preserves the following multisymplectic conservation law  

<|ref|>equation<|/ref|><|det|>[[112, 510, 930, 576]]<|/det|>
\[\begin{array}{rl} & {\partial_{t}\big[d\mathbf{E}_{i,j,k}^{l}\wedge d\mathbf{H}_{i,j,k}^{l}\big] + \partial_{x_{1}}\big[\frac{1}{\epsilon} d\mathbf{H}_{i - 1,j,k}^{l + 1}\wedge R_{1}d\mathbf{H}_{i,j,k}^{l + 1} + \frac{1}{\mu} d\mathbf{E}_{i - 1,j,k}^{l + 1}\wedge R_{1}d\mathbf{E}_{i,j,k}^{l + 1}\big] + \partial_{x_{2}}\big[\frac{1}{\epsilon} d\mathbf{H}_{i,j - 1,k}^{l + 1}\wedge R_{2}d\mathbf{H}_{i,j,k}^{l + 1} + \frac{1}{\mu} d\mathbf{E}_{i,j - 1,k}^{l + 1}\wedge R_{2}d\mathbf{E}_{i,j,k}^{l + 1}\big]}\\ & {\quad +\partial_{x_{3}}\big[\frac{1}{\epsilon} d\mathbf{H}_{i,j,k - 1}^{l + 1}\wedge R_{3}d\mathbf{H}_{i,j,k}^{l + 1} + \frac{1}{\mu} d\mathbf{E}_{i,j,k - 1}^{l + 1}\wedge R_{3}d\mathbf{E}_{i,j,k}^{l + 1}\big] = 0.} \end{array} \quad (21)\]  

<|ref|>text<|/ref|><|det|>[[72, 580, 930, 610]]<|/det|>
Besides the global conservation laws, for the scheme (18) applied to Maxwell's equations, we also have the following local conservation laws based on (13)- (15):  

<|ref|>text<|/ref|><|det|>[[93, 610, 393, 624]]<|/det|>
The discrete quadratic conservation law is  

<|ref|>equation<|/ref|><|det|>[[112, 628, 930, 688]]<|/det|>
\[\begin{array}{rl} & {\frac{1}{2}\partial_{t}\big[\mu \mathbf{H}_{i,j,k}^{l}\cdot \mathbf{H}_{i,j,k}^{l} + \epsilon \mathbf{E}_{i,j,k}^{l}\cdot \mathbf{E}_{i,j,k}^{l}\big] + \frac{1}{2}\partial_{x_{1}}\big[\mathbf{H}_{i,j,k}^{l + 1}\cdot R_{1}\mathbf{E}_{i - 1,j,k}^{l + 1} + \mathbf{H}_{i - 1,j,k}^{l + 1}\cdot R_{1}\mathbf{E}_{i,j,k}^{l + 1}\big] + \frac{1}{2}\partial_{x_{2}}\big[\mathbf{H}_{i,j,k}^{l + 1}\cdot R_{2}\mathbf{E}_{i,j,k}^{l + 1} + \mathbf{H}_{i,j,k}^{l + 1}\cdot R_{2}\mathbf{E}_{i,j,k}^{l + 1}\big]}\\ & {\quad +\frac{1}{2}\partial_{x_{3}}\big[\mathbf{H}_{i,j,k}^{l + 1}\cdot R_{3}\mathbf{E}_{i,j,k}^{l + 1} + \mathbf{H}_{i,j,k}^{l + 1}\cdot R_{3}\mathbf{E}_{i,j,k}^{l + 1}\big] = 0.} \end{array} \quad (22)\]  

<|ref|>text<|/ref|><|det|>[[72, 692, 360, 707]]<|/det|>
The discrete energy conservation law is  

<|ref|>equation<|/ref|><|det|>[[112, 710, 930, 779]]<|/det|>
\[\begin{array}{rl} & {\partial_{t}\Big[\frac{1}{2\epsilon}\mathbf{H}_{i,j,k}^{l}\cdot \widehat{\nabla}\times \mathbf{H}_{i,j,k}^{l} + \frac{1}{2\mu}\mathbf{E}_{i,j,k}^{l}\cdot \widehat{\nabla}\times \mathbf{E}_{i,j,k}^{l}\Big] + \partial_{x_{1}}\Big[\frac{1}{2\epsilon}\partial_{t}\mathbf{H}_{i,j,k}^{l}\cdot R_{1}\mathbf{H}_{i - 1,j,k}^{l + 1} + \frac{1}{2\mu}\partial_{t}\mathbf{E}_{i,j,k}^{l}\cdot R_{1}\mathbf{E}_{i - 1,j,k}^{l + 1}\Big]}\\ & {\quad +\partial_{x_{2}}\Big[\frac{1}{2\epsilon}\partial_{t}\mathbf{H}_{i,j,k}^{l}\cdot R_{2}\mathbf{H}_{i,j,k}^{l + 1} + \frac{1}{2\mu}\partial_{t}\mathbf{E}_{i,j,k}^{l}\cdot R_{2}\mathbf{E}_{i,j,k}^{l + 1}\Big] + \partial_{x_{3}}\Big[\frac{1}{2\epsilon}\partial_{t}\mathbf{H}_{i,j,k}^{l}\cdot R_{3}\mathbf{H}_{i,j,k}^{l + 1} + \frac{1}{2\mu}\partial_{t}\mathbf{E}_{i,j,k}^{l}\cdot R_{3}\mathbf{E}_{i,j,k}^{l + 1}\Big] = 0.} \end{array} \quad (23)\]  

<|ref|>text<|/ref|><|det|>[[72, 782, 388, 797]]<|/det|>
The discrete momentum conservation law is