<table>
  <tr>
    <th>5</th><th>3</th><th>4</th><th>6</th><th>7</th><th>8</th><th>9</th><th>1</th><th>2</th>
  </tr>
  <tr>
    <td>6</td><td>7</td><td>2</td><td>1</td><td>9</td><td>5</td><td>3</td><td>4</td><td>8</td>
  </tr>
  <tr>
    <td>1</td><td>9</td><td>8</td><td>3</td><td>4</td><td>2</td><td>5</td><td>6</td><td>7</td>
  </tr>
  <tr>
    <td>8</td><td>5</td><td>9</td><td>7</td><td>6</td><td>1</td><td>4</td><td>2</td><td>3</td>
  </tr>
  <tr>
    <td>4</td><td>2</td><td>6</td><td>8</td><td>5</td><td>3</td><td>7</td><td>9</td><td>1</td>
  </tr>
  <tr>
    <td>7</td><td>1</td><td>3</td><td>9</td><td>2</td><td>4</td><td>8</td><td>5</td><td>6</td>
  </tr>
  <tr>
    <td>9</td><td>6</td><td>1</td><td>5</td><td>3</td><td>7</td><td>2</td><td>8</td><td>4</td>
  </tr>
  <tr>
    <td>2</td><td>8</td><td>7</td><td>4</td><td>1</td><td>9</td><td>6</td><td>3</td><td>5</td>
  </tr>
  <tr>
    <td>3</td><td>4</td><td>5</td><td>2</td><td>8</td><td>6</td><td>1</td><td>7</td><td>9</td>
  </tr>
</table>

Fig. 10.3 Solved sudoku puzzle.

10.4 Hybrid Optimization

Hybrid methods may be required to solve particularly difficult real-world optimization problems. Implementation of hybrid methods typically requires non-trivial scripting because hybrids generally require a custom optimization workflow process. To illustrate this point, we describe a hybrid optimization method to solve a parameter estimation problem arising in the context of a model for childhood infectious disease transmission.

This parameter estimation model is a difficult nonconvex, nonlinear program for which no efficient solution algorithm exists to find global optima. To facilitate solution, the problem is reformulated using a MIP under-estimator and an NLP over-estimator. Information is exchanged between the two formulations, and the process is iterated until the two solutions converge.

Example 10.A.14 provides a Pyomo script that implements the hybrid search. Note that some initialization code is omitted from this example for clarity. In this example, user-defined Python functions are used to organize and modularize the optimization workflow. For example, the optimization model in this application is constructed via a function, disease_mdl. Example 10.A.14 includes a sketch of this function.

Beyond the standard Pyomo model constructs, we observe the ability to dynamically augment Pyomo models with arbitrary data, e.g., the definition of the pts_Ls and pts_LI attributes; these are not model components, but rather raw Python data types. Pyomo itself is unaware of of these attributes, but other components of a user-defined script can access and manipulate these attributes. Such a mechanism is invaluable when information is being propagated across disparate components of a