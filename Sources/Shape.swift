public struct Shape {
    public let dimensions: [Dimension]
    
    public func volume() -> Int {
        return dimensions.reduce(1) { $0 * $1.value }
    }
    
    public init(_ dimensions: [Dimension]) {
        self.dimensions = dimensions
    }
}

extension Shape: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Dimension...) {
        self.init(elements)
    }
}

extension Shape: Equatable {}
public func ==(lhs: Shape, rhs: Shape) -> Bool {
    return lhs.dimensions == rhs.dimensions
}

extension Shape: CustomStringConvertible {
    public var description: String {
        return dimensions.description
    }
}
