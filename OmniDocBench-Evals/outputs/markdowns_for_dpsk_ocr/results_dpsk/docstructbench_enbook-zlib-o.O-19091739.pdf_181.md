<|ref|>text<|/ref|><|det|>[[105, 75, 707, 140]]<|/det|>
Model We are asked to find the magnetic field due to a simple current distribution, so this example is a typical problem for which the Biot- Savart law is appropriate. We must find the field contribution from a small element of current and then integrate over the current distribution from \(\theta_{1}\) to \(\theta_{2}\) , as shown in Figure 29.3b.  

<|ref|>text<|/ref|><|det|>[[105, 145, 715, 255]]<|/det|>
Analyse Let's start by considering a length element \(d\vec{s}\) located a distance \(r\) from \(P\) . The direction of the magnetic field at point \(P\) due to the current in this element is out of the page because \(d\vec{s}\times \hat{\mathbf{r}}\) is out of the page. In fact, because all the current elements \(Id\vec{s}\) lie in the plane of the page, they all produce a magnetic field directed out of the page at point \(P\) . Therefore, the direction of the magnetic field at point \(P\) is out of the page and we need only find the magnitude of the field. We place the origin at \(O\) and let point \(P\) be along the positive \(y\) axis, with \(\hat{\mathbf{k}}\) being a unit vector pointing out of the page.  

<|ref|>text<|/ref|><|det|>[[105, 259, 715, 308]]<|/det|>
From the geometry in Figure 29.3a, we can see that the angle between the vectors \(d\vec{s}\) and \(\vec{\mathbf{r}}\) is \(\left(\frac{\pi}{2} -\theta\right)\) radians.  

<|ref|>text<|/ref|><|det|>[[106, 312, 437, 328]]<|/det|>
Evaluate the cross product in the Biot- Savart law:  

<|ref|>equation<|/ref|><|det|>[[252, 338, 622, 377]]<|/det|>
\[d\vec{s}\times \hat{\mathbf{r}} = \left|d\vec{s}\times \hat{\mathbf{r}}\right|\hat{\mathbf{k}} = \left[d\vec{s}\sin \left(\frac{\pi}{2} -\theta\right)\right]\hat{\mathbf{k}} = (d\vec{x}\cos \theta)\hat{\mathbf{k}}\]  

<|ref|>text<|/ref|><|det|>[[106, 384, 305, 400]]<|/det|>
Substitute into Equation 29.1:  

<|ref|>equation<|/ref|><|det|>[[339, 408, 560, 443]]<|/det|>
\[d\vec{B} = (dB)\hat{\mathbf{k}} = \frac{\mu_0I}{4\pi}\frac{dx\cos\theta}{r^2}\hat{\mathbf{k}}\]  

<|ref|>text<|/ref|><|det|>[[106, 450, 500, 468]]<|/det|>
From the geometry in Figure 29.3a, express \(r\) in terms of \(\theta\) :  

<|ref|>equation<|/ref|><|det|>[[406, 475, 710, 508]]<|/det|>
\[r = \frac{a}{\cos\theta} \quad (2)\]  

<|ref|>image<|/ref|><|det|>[[732, 81, 916, 380]]<|/det|>
<|ref|>image_caption<|/ref|><|det|>[[732, 392, 912, 474]]<|/det|>
<center>Figure 29.3 (Example 29.1) (a) A thin, straight wire carrying a current \(I\) (b) The angles \(\theta_{1}\) and \(\theta_{2}\) are used for determining the net field. </center>  

<|ref|>text<|/ref|><|det|>[[105, 515, 912, 549]]<|/det|>
Notice that \(\tan \theta = - x / a\) from the right triangle in Figure 29.3a (the negative sign is necessary because \(d\vec{s}\) is located at a negative value of \(x\) ) and solve for \(x\) :  

<|ref|>equation<|/ref|><|det|>[[466, 556, 558, 572]]<|/det|>
\[x = -a\tan \theta\]  

<|ref|>text<|/ref|><|det|>[[106, 580, 265, 595]]<|/det|>
Find the differential \(dx\) :  

<|ref|>equation<|/ref|><|det|>[[421, 600, 911, 636]]<|/det|>
\[dx = -a\sec^2\theta d\theta = -\frac{a d\theta}{\cos^2\theta} \quad (3)\]  

<|ref|>text<|/ref|><|det|>[[105, 648, 642, 666]]<|/det|>
Substitute Equations (2) and (3) into the magnitude of the field from Equation (1):  

<|ref|>equation<|/ref|><|det|>[[340, 672, 911, 712]]<|/det|>
\[dB = -\frac{\mu_0I}{4\pi}\left(\frac{a d\theta}{\cos^2\theta}\right)\left(\frac{\cos^2\theta}{a^2}\right)\cos \theta = -\frac{\mu_0I}{4\pi a}\cos \theta d\theta \quad (4)\]  

<|ref|>text<|/ref|><|det|>[[105, 722, 904, 755]]<|/det|>
Integrate Equation (4) over all length elements on the wire, where the subtending angles range from \(\theta_{1}\) to \(\theta_{2}\) as defined in Figure 29.3b:  

<|ref|>equation<|/ref|><|det|>[[360, 760, 911, 797]]<|/det|>
\[B = -\frac{\mu_0I}{4\pi a}\int_{\theta_1}^{\theta_2}\cos \theta d\theta = \frac{\mu_0I}{4\pi a} (\sin \theta_1 - \sin \theta_2) \quad (29.4)\]  

<|ref|>text<|/ref|><|det|>[[105, 805, 610, 822]]<|/det|>
Check the dimensions, noting that the quantity in brackets is dimensionless:  

<|ref|>equation<|/ref|><|det|>[[350, 826, 670, 847]]<|/det|>
\[[\mathrm{MQ}^{-1}\mathrm{T}^{-1}] = [\mathrm{MLQ}^{-2}][\mathrm{QT}^{-1}] / [\mathrm{L}] = [\mathrm{MQ}^{-1}\mathrm{T}^{-1}]\circledast\]  

<|ref|>text<|/ref|><|det|>[[105, 850, 662, 868]]<|/det|>
(B) Find an expression for the field at a point near a very long current-carrying wire.  

<|ref|>sub_title<|/ref|><|det|>[[105, 884, 184, 900]]<|/det|>
## Solution  

<|ref|>text<|/ref|><|det|>[[105, 900, 894, 934]]<|/det|>
We can use Equation 29.4 to find the magnetic field of any straight current- carrying wire if we know the geometry and hence the angles \(\theta_{1}\) and \(\theta_{2}\) . If the wire in Figure 29.3b becomes infinitely long, we see that \(\theta_{1} = \pi /2\) and \(\theta_{2} = -\pi /2\) for