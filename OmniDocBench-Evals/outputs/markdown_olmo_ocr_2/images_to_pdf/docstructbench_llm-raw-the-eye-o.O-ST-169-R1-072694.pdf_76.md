Result Parameter Details

Peripheral Control Status
The SMPC outputs the peripheral control status to the status register (SR) when the SMPC control mode is used. The status register (SR) is a register that can be read without regard for the INTBACK command. However, when the register is read when the INTBACK command is not in use, all bits except the RESB bit will be undefined.

<table>
  <tr>
    <th>bit7</th>
    <th>1</th>
    <th>PDL</th>
    <th>NPE</th>
    <th>RESB</th>
    <th>P2MD1</th>
    <th>P2MD0</th>
    <th>P1MD1</th>
    <th>P1MD0</th>
    <th>bit0</th>
  </tr>
  <tr>
    <td colspan="9">SR</td>
  </tr>
  <tr>
    <td colspan="9">
      P1MD: Port 1 Mode<br>
      00: 15-byte mode (Returns peripheral data up to a maximum of 15 bytes.)<br>
      01: 255-byte mode (Returns peripheral data up to a maximum of 255 bytes.)<br>
      10: Unused<br>
      11: 0-byte mode (Port is not accessed.)
    </td>
  </tr>
  <tr>
    <td colspan="9">
      P2MD: Port 2 Mode<br>
      00: 15-byte mode (Returns peripheral data up to a maximum of 15 bytes.)<br>
      01: 255-byte mode (Returns peripheral data up to a maximum of 255 bytes.)<br>
      10: Unused<br>
      11: 0-byte mode (Port is not accessed.)
    </td>
  </tr>
  <tr>
    <td colspan="9">
      RESB: Reset Button Status Bit<br>
      0: Reset Button OFF<br>
      1: Reset Button ON<br>
      Reading without regard for INTBACK command is possible. (Shows status for each V-BLANK-IN.)
    </td>
  </tr>
  <tr>
    <td colspan="9">
      NPE: Remaining Peripheral Existence Bit<br>
      0: No remaining data<br>
      1: Remaining data
    </td>
  </tr>
  <tr>
    <td colspan="9">
      PDL: Peripheral Data Location Bit<br>
      0: 2nd or above peripheral data<br>
      1: 1st peripheral data
    </td>
  </tr>
  <tr>
    <td colspan="9">
      bit7: Always 1
    </td>
  </tr>
</table>

Figure 3.13 Peripheral Control Status