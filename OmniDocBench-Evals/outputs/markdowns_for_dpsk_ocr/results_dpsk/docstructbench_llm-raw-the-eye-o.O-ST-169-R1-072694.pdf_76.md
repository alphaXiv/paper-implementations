<|ref|>sub_title<|/ref|><|det|>[[86, 65, 320, 82]]<|/det|>
## Result Parameter Details  

<|ref|>sub_title<|/ref|><|det|>[[86, 89, 286, 104]]<|/det|>
## Peripheral Control Status  

<|ref|>text<|/ref|><|det|>[[85, 105, 812, 195]]<|/det|>
The SMPC outputs the peripheral control status to the status register (SR) when the SMPC control mode is used. The status register (SR) is a register that can be read without regard for the INTBACK command. However, when the register is read when the INTBACK command is not in use, all bits except the RESB bit will be undefined.  

<|ref|>table<|/ref|><|det|>[[85, 240, 828, 635]]<|/det|>

<table><tr><td rowspan="2">SR</td><td colspan="6">bit7</td><td colspan="3">bit0</td></tr><tr><td>1</td><td>PDL</td><td>NPE</td><td>RESB</td><td>P2MD1</td><td>P2MD0</td><td>P1MD1</td><td>P1MD0</td><td></td></tr><tr><td></td><td colspan="8">P1MD: Port 1 Mode<br>00: 15-byte mode (Returns peripheral data up to a maximum of 15 bytes.)<br>01: 255-byte mode (Returns peripheral data up to a maximum of 255 bytes.)<br>10: Unused<br>11: 0-byte mode (Port is not accessed.)</td><td></td></tr><tr><td></td><td colspan="8">P2MD: Port 2 Mode<br>00: 15-byte mode (Returns peripheral data up to a maximum of 15 bytes.)<br>01: 255-byte mode (Returns peripheral data up to a maximum of 255 bytes.)<br>10: Unused<br>11: 0-byte mode (Port is not accessed.)</td><td></td></tr><tr><td></td><td colspan="8">RESB: Reset Button Status Bit<br>0: Reset Button OFF<br>1: Reset Button ON<br>Reading without regard for INTBACK command is possible. (Shows status for each V-BLANK-IN.)</td><td></td></tr><tr><td></td><td colspan="8">NPE: Remaining Peripheral Existence Bit<br>0: No remaining data<br>1: Remaining data</td><td></td></tr><tr><td></td><td colspan="8">PDL: Peripheral Data Location Bit<br>0: 2nd or above peripheral data<br>1: 1st peripheral data</td><td></td></tr><tr><td></td><td colspan="8">bit7: Always 1</td><td></td></tr></table>  

<|ref|>image<|/ref|><|det|>[[115, 655, 618, 900]]<|/det|>
<|ref|>image_caption<|/ref|><|det|>[[320, 673, 617, 690]]<|/det|>
<center>Figure 3.13 Peripheral Control Status </center>