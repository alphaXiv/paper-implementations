Now let's review these concepts with the following tree example:

![A diagram showing a tree data structure with labeled root, nodes, leaves, edge, and node.](page_374_256_800_384.png)

Tree data structure elements

• Root node: [2].
• Nodes: [2, 8, 1, 7, 11, 5].
• Leaves: [7, 11, 5].
• Height: There are two edges between the root [2] and the furthest connected leaf (which could be [7], [11], or [5] with same distance from the root). So the height of the tree is 2.
• Parent example: [8] is the parent of [7, 11].
• Children example: [7, 11] are the children of [8]. [5] is the child of [1].
• Subtrees: Starting in the root node [2], It has two subtrees: one is [8, 7, 11] and another one is [1, 5].
• Height of node [8]: 1.
• Depth of node [5]: 2.
• Level of root node: Depth + 1 = 0 + 1 = 1.

Overview of different types of tree

There are different types of tree data structures, each one with their own benefits and implementations. We are going to have a quick look over the most common ones so we can gain a good overview of the different types and choose which one to use in each case wisely.

After the following introduction to the different types of trees, we will go deeper into the details, properties, uses, and implementations.