The sink and swim methods were inspired by the book Algorithms by Sedgewick and Wayne, Fourth Edition, Section 2.4.

The structure accepts any type that conforms to the Comparable protocol. The single initializer allows you to optionally specify the sorting order and a list of starting values. The default sorting order is descending and the default starting values are an empty collection:

    /// Initialization
    var priorityQueue = PriorityQueue<String>(ascending: true)

    /// Initializing with starting values
    priorityQueue = PriorityQueue<String>(ascending: true, startingValues: ["Coldplay", "OneRepublic", "Maroon 5", "Imagine Dragons", "The Script"])

    var x = priorityQueue.pop()
    /// Coldplay

    x = priorityQueue.pop()
    /// Imagine Dragons

Protocols

The PriorityQueue conforms to sequence, collection, and IteratorProtocol, so you can treat it like any other Swift sequence and collection:

    extension PriorityQueue: IteratorProtocol {
        public typealias Element = T
        mutating public func next() -> Element? { return pop() }
    }

    extension PriorityQueue: Sequence {
        public typealias Iterator = PriorityQueue
        public func makeIterator() -> Iterator { return self }
    }

This allows you to use Swift standard library functions on a PriorityQueue and iterate through a PriorityQueue like this:

    for x in priorityQueue {
        print(x)
    }

    // Coldplay