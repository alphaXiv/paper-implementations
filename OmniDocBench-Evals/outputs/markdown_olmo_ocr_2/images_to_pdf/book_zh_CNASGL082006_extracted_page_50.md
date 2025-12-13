\( P_x = \frac{w}{t} \)      \( w = \frac{n}{N} \times 3.6 \times 10^6 \) 瓦秒

\( P_x \)——被测功率 W;
\( w \)——电度表累积测得电能;
\( N \)——电度表每千瓦时盘转动数;
\( n \)——电度表的转数 转;
\( t \)——测量时间 s

3 方差与传播系数

\[
u^2(P_x) = \left( \frac{\partial f}{\partial w} \right)^2 u^2(w) + \left( \frac{\partial f}{\partial t} \right)^2 u^2(t) = c^2(w)u^2(w) + c^2(t)u^2(t)
\]

\[
c(w) = \frac{1}{t} \qquad c(t) = -\frac{w}{t^2}
\]

\[
u^2(P_x) = \frac{1}{t^2} u^2(w) + \frac{w^2}{t^4} u^2(t)
\]

\[
\left( \frac{u(P_x)}{P_x} \right)^2 = \frac{u^2(w)}{w^2} + \frac{u^2(t)}{t^2}
\]

本不确定度分析以榨汁机为例

4 标准不确定度一览表

<table>
  <tr>
    <th>标准不确定度分量 \( u_i \)</th>
    <th>不确定度来源</th>
    <th>标准不确定度值</th>
    <th>\( c_i = \frac{\partial f}{\partial x_i} \)</th>
    <th>\( |c_i| \times u(x_i) \)</th>
    <th>自由度</th>
  </tr>
  <tr>
    <td>\( u_1 \)</td>
    <td>重复性误差</td>
    <td>0.2%</td>
    <td>1</td>
    <td>0.2%</td>
    <td>2</td>
  </tr>
  <tr>
    <td>\( u_2 \)</td>
    <td>表头示值误差</td>
    <td>0.29%</td>
    <td>1</td>
    <td>0.29%</td>
    <td>50</td>
  </tr>
  <tr>
    <td>\( u_3 \)</td>
    <td>电子秒表误差</td>
    <td>0.02%</td>
    <td>1</td>
    <td>0.02%</td>
    <td>8</td>
  </tr>
</table>

\[
u_c(P_x) = 0.35\%
\]
\[
v_{eff} = 16
\]