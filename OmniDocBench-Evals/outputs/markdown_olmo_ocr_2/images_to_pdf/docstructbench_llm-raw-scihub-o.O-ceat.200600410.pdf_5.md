Table 1. Considered process variables of the FCCU case study.

<table>
  <tr>
    <th>Variable</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Flow of wash oil to reactor riser</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Flow of fresh feed to reactor riser</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Flow of slurry to reactor riser</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Temperature of fresh feed entering furnace</td>
  </tr>
  <tr>
    <td>5</td>
    <td>Fresh feed temperature to riser</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Furnace firebox temperature</td>
  </tr>
  <tr>
    <td>7</td>
    <td>Combustion air blower inlet suction flow</td>
  </tr>
  <tr>
    <td>8</td>
    <td>Combustion air blower throughput</td>
  </tr>
  <tr>
    <td>9</td>
    <td>Combustion air flow rate</td>
  </tr>
  <tr>
    <td>10</td>
    <td>Lift air blower suction flow</td>
  </tr>
  <tr>
    <td>11</td>
    <td>Lift air blower speed</td>
  </tr>
  <tr>
    <td>12</td>
    <td>Lift air blower throughput</td>
  </tr>
  <tr>
    <td>13</td>
    <td>Riser temperature</td>
  </tr>
  <tr>
    <td>14</td>
    <td>Wet gas compressor suction pressure</td>
  </tr>
  <tr>
    <td>15</td>
    <td>Wet gas compressor inlet suction flow</td>
  </tr>
  <tr>
    <td>16</td>
    <td>Wet gas flow to vapor recovery unit</td>
  </tr>
  <tr>
    <td>17</td>
    <td>Regenerator bed temperature</td>
  </tr>
  <tr>
    <td>18</td>
    <td>Stack gas valve position</td>
  </tr>
  <tr>
    <td>19</td>
    <td>Regenerator pressure</td>
  </tr>
  <tr>
    <td>20</td>
    <td>Standpipe catalyst level</td>
  </tr>
  <tr>
    <td>21</td>
    <td>Stack gas O<sub>2</sub> concentration</td>
  </tr>
  <tr>
    <td>22</td>
    <td>Combustion air blower discharge pressure</td>
  </tr>
  <tr>
    <td>23</td>
    <td>Wet gas composition suction valve position</td>
  </tr>
</table>

Figure 2. Scatter plot in original feature space.

![Scatter plot in original feature space.](page_104_1042_627_496.png)

Figure 3. Scatter plot in high-dimensional feature space.

![Scatter plot in high-dimensional feature space.](page_728_104_627_496.png)

dataset, which is generated from simulation studies, in the FCCU process and extracted from the first and second optimal discriminant vector using Fisher discriminant analysis. Then the data was projected to the optimal discriminant vector, which resulted in the generation of a scatter plot of the first and second feature vector in the original space. It can be seen in Fig. 2 that only fault 1 can be differentiated clearly from the normal data and that faults 2 and 3 cannot be differentiated from normal data. The reason for this is that FDA is a linear method in operation. Consequently, it has a poor ability to deal with data which shows complex nonlinear relationship among variables. The scatter plot of the first kernel Fisher feature vector and the second vector via kernel FDA is presented in Fig. 3. It is seen from Fig. 3 that after projecting to the high-dimensional feature space through selecting the appropriate kernel function, the kernel Fisher discriminant method can easily discriminate data that belong to different classes.

The RBF function is used as the selected kernel function, and the parameter \( c \) is selected as 0.8 according to experience, viz:

\[
K(x_i, x_j) = \exp(-\frac{\|x_i - x_j\|^2}{c})
\]

(22)

The process disturbances considered are listed in Tab. 2. A 10 % loss of combustion air blower capacity was selected for

Table 2. Process disturbances for FCCU.

<table>
  <tr>
    <th>Case</th>
    <th>Disturbance</th>
  </tr>
  <tr>
    <td>1</td>
    <td>10 % loss of combustion air blower capacity</td>
  </tr>
  <tr>
    <td>2</td>
    <td>5 % degradation in the flow of regenerated catalyst</td>
  </tr>
  <tr>
    <td>3</td>
    <td>5 % increase in the coke factor of the feed</td>
  </tr>
  <tr>
    <td>4</td>
    <td>10 % decrease in the heat exchanger coefficient of the furnace</td>
  </tr>
  <tr>
    <td>5</td>
    <td>10 % increase in fresh feed</td>
  </tr>
  <tr>
    <td>6</td>
    <td>5 % decrease in lift air blower speed</td>
  </tr>
  <tr>
    <td>7</td>
    <td>5 % increase in friction coefficient of regenerated catalyst</td>
  </tr>
  <tr>
    <td>8</td>
    <td>Negative bias of reactor pressure sensor</td>
  </tr>
</table>