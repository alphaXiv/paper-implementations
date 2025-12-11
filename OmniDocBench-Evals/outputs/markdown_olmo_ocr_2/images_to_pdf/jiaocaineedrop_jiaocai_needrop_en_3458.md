解法4：\( F(x) = \frac{g(x)}{f(x)} = \frac{\sin(2x + \frac{\pi}{4})}{\sin(2x - \frac{\pi}{4})} \)，定义域为 \( \left\{ x \mid x \neq \frac{\pi}{8} + \frac{k\pi}{2}, k \in \mathbb{Z} \right\} \).

\[
F'(x) = \frac{2\cos(2x + \frac{\pi}{4})\sin(2x - \frac{\pi}{4}) - 2\cos(2x - \frac{\pi}{4})\sin(2x + \frac{\pi}{4})}{\sin^2(2x - \frac{\pi}{4})} = \frac{-2}{\sin^2(2x - \frac{\pi}{4})} < 0
\]

\( \therefore F(x) \) 的单调递减区间为 \( (-\frac{3\pi}{8} + \frac{k\pi}{2}, \frac{\pi}{8} + \frac{k\pi}{2}) \), \( k \in \mathbb{Z} \)，无递增区间.

评分标准：
(1) \( g(x) \) 表达式正确2分，
(2) 有化简结果正确2分：\( -\tan(2x + \frac{\pi}{4}) \cdot \frac{1}{\tan(2x - \frac{\pi}{4})} \cdot \frac{1 + \frac{2}{\tan 2x - 1}}{\sin^2(2x - \frac{\pi}{4})} < 0 \)
(3) 单调区间正确1分。

注：1. 有正确的体积公式，h没算对，这2分也给！
2. 没有写出以上的任何一个踩分点，去证明A1E垂直与平面ABCD的，给2分

18. 解：（1）连接AD1，因为AE//DC1，所以A,E,C1,D1四点共面，
因为C1E//平面ADD1A1，AD1是过C1E的平面AEC1D1与平面ADD1A1的交线
由面面平行的性质定理，知AD1//C1E
所以四边形AEC1D1为平行四边形 2分
所以 \( AE = D_1C_1 = \frac{1}{2}DC = \frac{1}{2} \)
易得 \( \angle AAE = 60^\circ \)，又 \( AA_1 = 1 \)，
所以 \( AE = \sqrt{AA_1^2 + AE^2 - 2AA_1 \cdot AE \cdot \cos 60^\circ} = \frac{\sqrt{3}}{2} \) 同时可得 \( AE \perp AB \).
上下底面积分别为 \( S_1, S_2 \)，易求得 \( S_1 = \frac{3}{4} \)，\( S_2 = \frac{3}{2} \)
所以 \( \frac{7\sqrt{3}}{16} = \frac{h}{3}(S_1 + \sqrt{S_1S_2} + S_2) = \frac{h}{3}(\frac{3}{4} + \frac{3}{2} + \frac{3}{2}) = \frac{7}{8}h \)，从而有 \( h = \frac{\sqrt{3}}{2} \) 2分
所以 \( h = AE \)

（2）解法1：由（1）知平面AA1B1B⊥平面ABCD
又 \( BC \perp AB \)，所以 \( BC \perp \) 平面AA1B1B
所以平面BCC1B1⊥平面AA1B1B 2分
过E作EH⊥BB1于H，则EH⊥平面BCC1B1
从而 \( \angle EC_1H \) 为直线C1E与平面BCC1B1所成角 2分
\[ EH = BE \sin 60^\circ = \frac{3\sqrt{3}}{4} \]
\[ C_1E^2 = C_1B_1^2 + B_1E^2 = C_1B_1^2 + BE^2 + BB_1^2 - 2 \cdot BB_1 \cdot BE \cdot \cos 60^\circ = 2 \]，即 \( C_1E = \sqrt{2} \)
所以 \( \sin \angle EC_1H = \frac{EH}{C_1E} = \frac{3\sqrt{6}}{8} \). 2分