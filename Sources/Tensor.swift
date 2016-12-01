
#if os(iOS) || os(OSX)
import Accelerate
#endif

public struct Tensor {
    public typealias Element = Float
    
    public let shape: Shape
    public fileprivate(set) var elements: [Element]
    
    public init(shape: Shape, elements: [Element]) {
        let volume = shape.volume()
        precondition(elements.count >= volume, "`elements.count` must be greater than or equal to `shape.volume`: elements.count = \(elements.count), shape.volume = \(shape.volume())")
        self.shape = shape
        self.elements = (elements.count == volume) ? elements : Array(elements[0..<volume])
    }
}

extension Tensor { // Additional Initializers
    public init(shape: Shape, element: Element = 0.0) {
        self.init(shape: shape, elements: [Element](repeating: element, count: shape.volume()))
    }
}

extension Tensor {
    public mutating func reshape(_ shape: Shape) {
        self = reshaped(shape)
    }
    
    public func reshaped(_ shape: Shape) -> Tensor {
        return Tensor(shape: shape, elements: elements)
    }
}

extension Tensor { // like CollentionType
    internal func index(_ indices: [Int]) -> Int {
        assert(indices.count == shape.dimensions.count, "`indices.count` must be \(shape.dimensions.count): \(indices.count)")
        return zip(shape.dimensions, indices).reduce(0) {
            assert(0 <= $1.1 && $1.1 < $1.0.value, "Illegal index: indices = \(indices), shape = \(shape)")
            return $0 * $1.0.value + $1.1
        }
    }
    
    public subscript(indices: Int...) -> Element {
        get {
            return elements[index(indices)]
        }
        set {
            elements[index(indices)] = newValue
        }
    }
    
    public func volume() -> Int {
        return shape.volume()
    }
}

extension Tensor: Sequence {
    public func makeIterator() -> IndexingIterator<[Element]> {
        return elements.makeIterator()
    }
}

extension Tensor: Equatable {}
public func ==(lhs: Tensor, rhs: Tensor) -> Bool {
    return lhs.shape == rhs.shape && lhs.elements == rhs.elements
}

internal func commutativeBinaryOperation(_ lhs: Tensor, _ rhs: Tensor, operation: (Float, Float) -> Float) -> Tensor {
    let lSize = lhs.shape.dimensions.count
    let rSize = rhs.shape.dimensions.count
    
    if lSize == rSize {
        precondition(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
        return Tensor(shape: lhs.shape, elements: zipMap(lhs.elements, rhs.elements, operation: operation))
    }
    
    let a: Tensor
    let b: Tensor
    if lSize < rSize {
        a = rhs
        b = lhs
    } else {
        a = lhs
        b = rhs
    }
    assert(hasSuffix(array: a.shape.dimensions, suffix: b.shape.dimensions), "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
    
    return Tensor(shape: a.shape, elements: zipMapRepeat(a.elements, b.elements, operation: operation))
}

internal func noncommutativeBinaryOperation(_ lhs: Tensor, _ rhs: Tensor, operation: (Float, Float) -> Float) -> Tensor {
    let lSize = lhs.shape.dimensions.count
    let rSize = rhs.shape.dimensions.count
    
    if lSize == rSize {
        precondition(lhs.shape == rhs.shape, "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
        return Tensor(shape: lhs.shape, elements: zipMap(lhs.elements, rhs.elements, operation: operation))
    } else if lSize < rSize {
        precondition(hasSuffix(array: rhs.shape.dimensions, suffix: lhs.shape.dimensions), "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
        return Tensor(shape: rhs.shape, elements: zipMapRepeat(rhs.elements, lhs.elements, operation: { operation($1, $0) }))
    } else {
        precondition(hasSuffix(array: lhs.shape.dimensions, suffix: rhs.shape.dimensions), "Incompatible shapes of tensors: lhs.shape = \(lhs.shape), rhs.shape = \(rhs.shape)")
        return Tensor(shape: lhs.shape, elements: zipMapRepeat(lhs.elements, rhs.elements, operation: operation))
    }
}

public func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    return commutativeBinaryOperation(lhs, rhs, operation: +)
}

public func -(lhs: Tensor, rhs: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(lhs, rhs, operation: -)
}

public func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    return commutativeBinaryOperation(lhs, rhs, operation: *)
}

public func /(lhs: Tensor, rhs: Tensor) -> Tensor {
    return noncommutativeBinaryOperation(lhs, rhs, operation: /)
}

public func *(lhs: Tensor, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { $0 * rhs })
}

public func *(lhs: Float, rhs: Tensor) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.elements.map { lhs * $0 })
}

public func /(lhs: Tensor, rhs: Float) -> Tensor {
    return Tensor(shape: lhs.shape, elements: lhs.elements.map { $0 / rhs })
}

public func /(lhs: Float, rhs: Tensor) -> Tensor {
    return Tensor(shape: rhs.shape, elements: rhs.elements.map { lhs / $0 })
}

extension Tensor { // Matrix
    public func matmul(_ tensor: Tensor) -> Tensor {
        precondition(shape.dimensions.count == 2, "This tensor is not a matrix: shape = \(shape)")
        precondition(tensor.shape.dimensions.count == 2, "The given tensor is not a matrix: shape = \(tensor.shape)")
        precondition(tensor.shape.dimensions[0] == shape.dimensions[1], "Incompatible shapes of matrices: self.shape = \(shape), tensor.shape = \(tensor.shape)")
        
        #if os(iOS) || os(OSX)
            let result = Tensor(shape: [shape.dimensions[0], tensor.shape.dimensions[1]])
            
            let n = Int32(tensor.shape.dimensions[1].value)
            let k = Int32(shape.dimensions[1].value)
            cblas_sgemm(
                CblasRowMajor,                                // Order
                CblasNoTrans,                                 // TransA
                CblasNoTrans,                                 // TransB
                Int32(shape.dimensions[0].value),             // M
                n,                                            // N
                k,                                            // K
                1.0,                                          // alpha
                elements,                                     // A
                k,                                            // lda
                tensor.elements,                              // B
                n,                                            // ldb
                1.0,                                          // beta
                UnsafeMutablePointer<Float>(mutating: result.elements), // C
                n                                             // ldc
            )
            
            return result
        #else
            let n = shape.dimensions[1].value
            
            let numRows = shape.dimensions[0]
            let numCols = tensor.shape.dimensions[1]
            
            let leftHead = UnsafeMutablePointer<Float>(self.elements)
            let rightHead = UnsafeMutablePointer<Float>(tensor.elements)
            
            let elements = [Float](count: (numCols * numRows).value, repeatedValue: 0.0)
            for r in 0..<numRows.value {
                for i in 0..<n {
                    var pointer = UnsafeMutablePointer<Float>(elements) + r * numCols.value
                    let left = leftHead[r * n + i]
                    var rightPointer = rightHead + i * numCols.value
                    for _ in 0..<numCols.value {
                        pointer.memory += left * rightPointer.memory
                        pointer += 1
                        rightPointer += 1
                    }
                }
            }
            
            return Tensor(shape: [numRows, numCols], elements: elements)
        #endif
    }
}
