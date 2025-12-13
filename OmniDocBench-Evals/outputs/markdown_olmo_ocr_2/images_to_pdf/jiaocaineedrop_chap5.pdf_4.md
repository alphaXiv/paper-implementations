(d)

<table>
  <tr>
    <th>Parsing stack</th>
    <th>Input</th>
    <th>Action</th>
  </tr>
  <tr>
    <td>1 $ 0</td>
    <td><b>int x,y,z$</b></td>
    <td>shift 3</td>
  </tr>
  <tr>
    <td>2 $ 0 int 3</td>
    <td>x,y,z$</td>
    <td>reduce 2</td>
  </tr>
  <tr>
    <td>3 $ 0 T 2</td>
    <td>x,y,z$</td>
    <td>shift 6</td>
  </tr>
  <tr>
    <td>4 $ 0 T 2 id 6</td>
    <td>,y,z$</td>
    <td>reduce 5</td>
  </tr>
  <tr>
    <td>5 $ 0 T 2 V 5</td>
    <td>,y,z$</td>
    <td>shift 7</td>
  </tr>
  <tr>
    <td>6 $ 0 T 2 V 5,7</td>
    <td>y,z$</td>
    <td>shift 8</td>
  </tr>
  <tr>
    <td>7 $ 0 T 2 V 5,7 id 8</td>
    <td>,z$</td>
    <td>reduce 4</td>
  </tr>
  <tr>
    <td>8 $ 0 T 2 V 5</td>
    <td>,z$</td>
    <td>shift 7</td>
  </tr>
  <tr>
    <td>9 $ 0 T 2 V 5,7</td>
    <td>z$</td>
    <td>shift 8</td>
  </tr>
  <tr>
    <td>10 $ 0 T 2 V 5,7 id 8</td>
    <td>$</td>
    <td>reduce 4</td>
  </tr>
  <tr>
    <td>11 $ 0 T 2 V 5</td>
    <td>$</td>
    <td>reduce 1</td>
  </tr>
  <tr>
    <td>12 $ 0 D 1</td>
    <td>$</td>
    <td>accept</td>
  </tr>
</table>

(e)

![A finite state automaton diagram showing parsing states and transitions for parsing integer and float types](page_320_624_1002_624.png)

(f) The LALR(1) parsing table is the same as the SLR(1) parsing table shown in (c).

Exercise 5.10

We use similar language to that on page 210, with appropriate modifications:

The SLR(1) parsing algorithm. Let s be the current state whose number is at the top of the parsing stack. Then actions are defined as follows:

(1) If state s contains any item of the form \( A \rightarrow \alpha . X \beta \), where \( X \) is a terminal, and \( X \) is the next token in the input string, then the action is to remove the current input token and push onto the stack the number of the state containing the item \( A \rightarrow \alpha X . \beta \).