- EUT 与频谱分析仪之间的链路 S 参数 0.724(m)，损耗 2.8dB(d)
则

\[
u_{EUT-\text{功分器}} = \frac{0.7 \times 0.099 \times 100}{\sqrt{2} \times 11.5} = 0.426\text{dB}
\]

\[
u_{\text{功分器}-\text{衰减网络}} = \frac{0.099 \times 0.333 \times 100}{\sqrt{2} \times 11.5} = 0.203\text{dB}
\]

\[
u_{\text{衰减网络}-\text{频谱仪}} = \frac{0.333 \times 0.091 \times 100}{\sqrt{2} \times 11.5} = 0.186\text{dB}
\]

\[
u_{EUT-\text{衰减网络}} = \frac{0.7 \times 0.333 \times 0.912 \times 0.912 \times 100}{\sqrt{2} \times 11.5} = 1.192\text{dB}
\]

\[
u_{\text{功分器}-\text{频谱仪}} = \frac{0.099 \times 0.333 \times 0.794 \times 0.794 \times 100}{\sqrt{2} \times 11.5} = 0.128\text{dB}
\]

\[
u_{EUT-\text{频谱仪}} = \frac{0.7 \times 0.091 \times 0.724 \times 0.724 \times 100}{\sqrt{2} \times 11.5} = 0.205\text{dB}
\]

则 \( u(\delta P_M) = \sqrt{0.426^2 + 0.203^2 + 0.186^2 + 1.192^2 + 0.128^2 + 0.205^2} = 1.318\text{dB} \)

5.5.3.5 被测样供电电压变化引入的不确定度分量 \( u(\delta P_v) \)

在测试期间，实验室供电电压可控范围为 0.1V，根据 ETSI TR 100 028 表 F.1
- 均值 10%(p)/V
- 标准差 3%(p)/V

\[
u(\delta P_v) = \frac{0.1V \times \sqrt{(10\% / V)^2 + (3\% / V)^2}}{\sqrt{3} \times 23.0} = 0.026\text{dB}
\]

5.5.3.6 时间周期变化引入的不确定度分量 \( u(\delta P_D) \)

根据 ETSI TR 100 028 表 F.1，时间周期误差为 2%(d)(p)(σ)

\[
u(\delta P_D) = \frac{2\%}{23.0} = 0.087\text{dB}
\]

5.5.4 不确定度概算

表 5-10 传导杂散发射测量不确定度概算表

<table>
  <tr>
    <th>分量</th>
    <th>概率分布</th>
    <th>灵敏系数</th>
    <th>不确定度分量值（dB）</th>
  </tr>
</table>