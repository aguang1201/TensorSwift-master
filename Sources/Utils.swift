extension Int {
    internal func ceilDiv(_ rhs: Int) -> Int {
        return (self + rhs - 1) / rhs
    }
}

internal func hasSuffix<Element: Equatable>(array: [Element], suffix: [Element]) -> Bool {
    guard array.count >= suffix.count else { return false }
    return zip(array[(array.count - suffix.count)..<array.count], suffix).reduce(true) { $0 && $1.0 == $1.1 }
}

internal func zipMap(_ a: [Float], _ b: [Float], operation: (Float, Float) -> Float) -> [Float] {
    var result: [Float] = []
    for i in a.indices {
        result.append(operation(a[i], b[i]))
    }
    return result
}

internal func zipMapRepeat(_ a: [Float], _ infiniteB: [Float], operation: (Float, Float) -> Float) -> [Float] {
    var result: [Float] = []
    for i in a.indices {
        result.append(operation(a[i], infiniteB[i % infiniteB.count]))
    }
    return result
}

