<|ref|>equation<|/ref|><|det|>[[88, 74, 485, 103]]<|/det|>
\[\Lambda_{\mathrm{r}}(\mathrm{r}) = \lambda_{\mathrm{bed}}(\mathrm{r}) + \mathrm{K}_{1}\mathrm{Pe}_{0}\frac{\mathrm{u}_{\mathrm{e}}}{\mathrm{u}_{0}}\mathrm{f}(\mathrm{R} - \mathrm{r})\lambda_{\mathrm{r}} \quad (17)\]  

<|ref|>text<|/ref|><|det|>[[89, 109, 488, 154]]<|/det|>
The damping function \(\mathrm{f}(\mathrm{R} - \mathrm{r})\) remains the same as for the mass transfer (Eq.(11)), while the slope parameter and the damping parameter are slightly modified to, respectively,  

<|ref|>equation<|/ref|><|det|>[[88, 160, 485, 191]]<|/det|>
\[\mathrm{K}_{1} = \frac{1}{8} \quad (18)\]  

<|ref|>equation<|/ref|><|det|>[[88, 202, 485, 235]]<|/det|>
\[\mathrm{K}_{2} = 0.44 + 4\exp \left(-\frac{\mathrm{Re}_{0}}{70}\right) \quad (19)\]  

<|ref|>text<|/ref|><|det|>[[90, 242, 488, 403]]<|/det|>
The heat release by adsorption, see [19] for \(\Delta \mathrm{H}_{\mathrm{ad}}\) is derived in the last term on the right- hand side of Eq.(14) from the change of solids load with time. This very term couples the energy with the mass balance, so that both have to be solved simultaneously in order to account for thermal effects. Heat transfer resistances to or in the particles are neglected. The terms \(\delta_{\mathrm{bed}}(\mathrm{r})\) and \(\lambda_{\mathrm{bed}}(\mathrm{r})\) in Eqs. (8), (10), (15) and (17) describe the isotropic effective diffusivity and thermal conductivity of the bed without fluid flow. Boundary and initial conditions for Eqs. (7) and (14) are recapitulated in Tab. 1.  

<|ref|>text<|/ref|><|det|>[[90, 404, 488, 475]]<|/det|>
On the basis of the above described general model various reductions are possible by neglecting thermal effects, the radial coordinate or gas- to- particle and intraparticle mass transfer resistances. From such reduced versions the following has been considered in more detail in the present work:  

<|ref|>text<|/ref|><|det|>[[90, 476, 488, 580]]<|/det|>
1) plug-flow model (1-D) with local equilibrium between the gas and the solids, 
2) plug-flow model (1-D) with mass transfer resistance to the solids, 
3) 2-D maldistribution model with local equilibrium, 
4) 2-D maldistribution model with mass transfer resistance to the solids.  

<|ref|>text<|/ref|><|det|>[[90, 581, 488, 655]]<|/det|>
In our terminology "plug flow" means that every influence of the radial coordinate is neglected, including the influence of the wall on porosity and flow velocity. However, axial dispersion, as expressed by the dispersion coefficient \(\mathrm{D}_{\mathrm{ax}}\) , is accounted for, so that the equation  

<|ref|>equation<|/ref|><|det|>[[89, 660, 488, 694]]<|/det|>
\[\overline{\psi}\frac{\partial Y}{\partial\mathrm{t}} = \mathrm{D}_{\mathrm{ax}}\frac{\partial^2Y}{\partial z^2} -\overline{\mathrm{u}}_0\frac{\partial Y}{\partial z} -[1 - \overline{\psi} ]\frac{\partial X}{\partial\mathrm{t}}\frac{\rho_{\mathrm{p}}}{\rho_{\mathrm{f}}} \quad (20)\]  

<|ref|>text<|/ref|><|det|>[[90, 699, 488, 758]]<|/det|>
applies to the isothermal plug flow models (models 1 and 2). Eq. (20) is the classical, conventional way to model packed bed adsorbers. Local equilibrium corresponds, in terms of the two- layers model from [19], to the limiting case of \(\beta_{\mathrm{f}}\to \infty\)  

<|ref|>text<|/ref|><|det|>[[510, 78, 907, 183]]<|/det|>
and \(\beta_{\mathrm{p}}\to \infty\) . At this limit, equilibrium is considered to be sufficient for calculating the response of the solid phase to changes of the concentration in the fluid. Model 4 is our complete, highest order model, as previously outlined and in exact correspondence to [13- 18]. Mainly this model has been evaluated for both isothermal and non- isothermal conditions.  

<|ref|>sub_title<|/ref|><|det|>[[510, 210, 825, 226]]<|/det|>
## 4 Numerical Solution and its Validation  

<|ref|>text<|/ref|><|det|>[[510, 242, 907, 434]]<|/det|>
The partial differential equation or equations of the various models have been solved by the method of lines. The numerical calculations were conducted for different mesh densities, and the results accepted when the change of calculated gas moisture content values was lower than \(0.05\%\) of the maximal difference of gas moisture content appearing in the packed bed. When the error was bigger, the mesh was made denser. Since the width of the concentration front is, in many cases, not much smaller than the length of the bed, equidistant meshes have been used in the axial direction. In the maldistribution models (models 3 and 4 in the previous section) meshes that were denser near the wall than in the center of the tube have been applied.  

<|ref|>text<|/ref|><|det|>[[510, 435, 907, 566]]<|/det|>
To check the numerical procedure, respective results have been compared with available analytical solutions. One of such a solution is attributed to Anzelius [1] and refers to model 2 after the classification of section 3, additionally reduced by neglecting axial dispersion \(\mathrm{D}_{\mathrm{ax}} = 0\) . Furthermore, it is assumed that the sorption equilibrium is throughout linear ("Henry's law"), and that the bed is long. The mass transfer resistance is attributed to the fluid phase. Then, axial profiles can be derived to  

<|ref|>equation<|/ref|><|det|>[[510, 572, 905, 604]]<|/det|>
\[\frac{C}{\mathrm{C}_{\mathrm{in}}} = \frac{1}{2}\mathrm{erfc}\left(\sqrt{\frac{\mathrm{e}}{\mathrm{e}}} -\sqrt{\mathrm{r}}\right) \quad (21)\]  

<|ref|>text<|/ref|><|det|>[[510, 610, 540, 622]]<|/det|>
with  

<|ref|>equation<|/ref|><|det|>[[510, 628, 905, 662]]<|/det|>
\[\xi = 6\frac{\beta_{\mathrm{f}}}{\mathrm{d}_{\mathrm{p}}\mathrm{u}}\frac{\mathrm{z}1 - \psi}{\psi} \quad (22)\]  

<|ref|>text<|/ref|><|det|>[[510, 666, 540, 679]]<|/det|>
and  

<|ref|>equation<|/ref|><|det|>[[510, 686, 905, 722]]<|/det|>
\[\tau = 6\frac{\beta_{\mathrm{f}}}{\mathrm{d}_{\mathrm{p}}\mathrm{K}}\left(\mathrm{t} - \frac{\mathrm{z}}{\mathrm{u}}\right) \quad (23)\]  

<|ref|>text<|/ref|><|det|>[[510, 728, 907, 758]]<|/det|>
In Eq. (21) the concentration of adsorbate in the gas phase, C, is used instead of the content, Y, assuming an ini  

<|ref|>table<|/ref|><|det|>[[89, 794, 907, 904]]<|/det|>
<|ref|>table_caption<|/ref|><|det|>[[91, 778, 353, 790]]<|/det|>
Table 1. Boundary and initial conditions for models.   

<table><tr><td rowspan="2">t &amp;gt; 0</td><td rowspan="2">0 ≤ r ≤ R</td><td rowspan="2">z = 0</td><td>Y = Yin or</td><td rowspan="2">T = Tin</td></tr><tr><td>u0(Yin - Y) = -Dax ∂Y/∂z</td></tr><tr><td></td><td></td><td>z = L</td><td>∂X/∂z = 0</td><td>∂Y/∂z = 0</td></tr><tr><td rowspan="2">t &amp;gt; 0</td><td rowspan="2">0 ≤ z ≤ L</td><td rowspan="2">r = 0</td><td>∂X/∂r = 0</td><td>∂Y/∂r = 0</td></tr><tr><td>∂X/∂r = 0</td><td>∂Y/∂r = 0</td></tr><tr><td>t = 0</td><td>0 ≤ r ≤ R</td><td>0 ≤ z ≤ L</td><td>X(r,z) = X0</td><td>Y(r,z) = Y0</td></tr></table>