public struct Dimension {
    public let value: Int
    
    public init(_ value: Int) {
        guard value >= 0 else { fatalError("`value` must be greater than or equal to 0: \(value)") }
        self.value = value
    }
}

extension Dimension: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int) {
        self.init(value)
    }
}

extension Dimension: Equatable {}
public func ==(lhs: Dimension, rhs: Dimension) -> Bool {
    return lhs.value == rhs.value
}

extension Dimension: CustomStringConvertible {
    public var description: String {
        return value.description
    }
}

public func +(lhs: Dimension, rhs: Dimension) -> Dimension {
    return Dimension(lhs.value + rhs.value)
}

public func -(lhs: Dimension, rhs: Dimension) -> Dimension {
    return Dimension(lhs.value - rhs.value)
}

public func *(lhs: Dimension, rhs: Dimension) -> Dimension {
    return Dimension(lhs.value * rhs.value)
}

public func /(lhs: Dimension, rhs: Dimension) -> Dimension {
    return Dimension(lhs.value / rhs.value)
}
