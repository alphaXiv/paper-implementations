Here’s an example. Suppose you have a table with 10 million rows, which uses a couple of gigabytes on disk. It has a VARCHAR(1000) column with the utf8 character set. This can use up to 3 bytes per character, for a worst-case size of 3,000 bytes. If you mention this column in your ORDER BY clause, a query against the whole table can require over 30 GB of temporary space just for the sort files!

If the Extra column of EXPLAIN contains “Using temporary,” the query uses an implicit temporary table.

Using ENUM instead of a string type

Sometimes you can use an ENUM column instead of conventional string types. An ENUM column can store a predefined set of distinct string values. MySQL stores them very compactly, packed into one or two bytes depending on the number of values in the list. It stores each value internally as an integer representing its position in the field definition list, and it keeps the “lookup table” that defines the number-to-string correspondence in the table’s .frm file. Here’s an example:

    mysql> CREATE TABLE enum_test(
    ->     e ENUM('fish','apple', 'dog') NOT NULL
    -> );
    mysql> INSERT INTO enum_test(e) VALUES('fish'), ('dog'), ('apple');

The three rows actually store integers, not strings. You can see the dual nature of the values by retrieving them in a numeric context:

    mysql> SELECT e + 0 FROM enum_test;
    +-------+
    | e + 0 |
    +-------+
    |   1   |
    |   3   |
    |   2   |
    +-------+

This duality can be terribly confusing if you specify numbers for your ENUM constants, as in ENUM('1', '2', '3'). We suggest you don’t do this.

Another surprise is that an ENUM field sorts by the internal integer values, not by the strings themselves:

    mysql> SELECT e FROM enum_test ORDER BY e;
    +-----+
    | e   |
    +-----+
    | fish|
    | apple|
    | dog |
    +-----+