<|ref|>text<|/ref|><|det|>[[114, 92, 144, 108]]<|/det|>
(d) 

<|ref|>table<|/ref|><|det|>[[198, 103, 780, 365]]<|/det|>
<table><tr><td></td><td>Parsing stack</td><td>Input</td><td>Action</td></tr><tr><td>1</td><td>\(0\)</td><td>int x,y,z</td><td>shift 3</td></tr><tr><td>2</td><td>\(0\) int 3</td><td>x,y,z</td><td>reduce 2</td></tr><tr><td>3</td><td>\(0\) T2</td><td>x,y,z</td><td>shift 6</td></tr><tr><td>4</td><td>\(0\) T2 id 6</td><td>, y,z</td><td>reduce 5</td></tr><tr><td>5</td><td>\(0\) T2 V5</td><td>, y,z</td><td>shift 7</td></tr><tr><td>6</td><td>\(0\) T2 V5,7</td><td>y,z</td><td>shift 8</td></tr><tr><td>7</td><td>\(0\) T2 V5,7 id 8</td><td>, z</td><td>reduce 4</td></tr><tr><td>8</td><td>\(0\) T2 V5</td><td>, z</td><td>shift 7</td></tr><tr><td>9</td><td>\(0\) T2 V5,7</td><td>z</td><td>shift 8</td></tr><tr><td>10</td><td>\(0\) T2 V5,7 id 8</td><td>\(\)</td><td>reduce 4</td></tr><tr><td>11</td><td>\(0\) T2 V5</td><td>\(\)</td><td>reduce 1</td></tr><tr><td>12</td><td>\(0\) D1</td><td>\(\)</td><td>accept</td></tr></table>

<|ref|>text<|/ref|><|det|>[[114, 381, 144, 397]]<|/det|>
(e) 

<|ref|>image<|/ref|><|det|>[[198, 393, 780, 680]]<|/det|>
 

<|ref|>text<|/ref|><|det|>[[114, 694, 778, 712]]<|/det|>
(f) The LALR(1) parsing table is the same as the SLR(1) parsing table shown in (c). 

<|ref|>sub_title<|/ref|><|det|>[[114, 747, 230, 764]]<|/det|>
Exercise 5.10 

<|ref|>text<|/ref|><|det|>[[114, 783, 725, 801]]<|/det|>
We use similar language to that on page 210, with appropriate modifications: 

<|ref|>text<|/ref|><|det|>[[114, 819, 833, 854]]<|/det|>
**The SLR(1) parsing algorithm.** Let \(s\) be the current state whose number is at the top of the parsing stack. Then actions are defined as follows: 

<|ref|>text<|/ref|><|det|>[[114, 856, 880, 910]]<|/det|>
(1) If state \(s\) contains any item of the form \(A \to \alpha.X\beta\), where \(X\) is a terminal, and \(X\) is the next token in the input string, then the action is to remove the current input token and push onto the stack the number of the state containing the item \(A \to \alpha.X.\beta\).