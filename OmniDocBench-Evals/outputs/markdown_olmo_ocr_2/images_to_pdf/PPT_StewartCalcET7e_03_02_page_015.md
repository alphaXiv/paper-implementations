The Quotient Rule

so

\[
\frac{d}{dx} \left( \frac{u}{v} \right) = \lim_{\Delta x \to 0} \frac{\Delta (u/v)}{\Delta x} = \lim_{\Delta x \to 0} \frac{v \frac{\Delta u}{\Delta x} - u \frac{\Delta v}{\Delta x}}{v(v + \Delta v)}
\]

As \( \Delta x \to 0, \Delta v \to 0 \) also, because \( v = g(x) \) is differentiable and therefore continuous.

Thus, using the Limit Laws, we get

\[
\frac{d}{dx} \left( \frac{u}{v} \right) = \frac{v \lim_{\Delta x \to 0} \frac{\Delta u}{\Delta x} - u \lim_{\Delta x \to 0} \frac{\Delta v}{\Delta x}}{v \lim_{\Delta x \to 0} (v + \Delta v)} = \frac{v \frac{du}{dx} - u \frac{dv}{dx}}{v^2}
\]