import XCTest
import TensorSwift

class TensorSwiftSample: XCTestCase {
    func testSample() {
        let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let b = Tensor(shape: [2, 3], elements: [7, 8, 9, 10, 11, 12])
        let sum = a + b // Tensor(shape: [2, 3], elements: [8, 10, 12, 14, 16, 18])
        let mul = a * b // Tensor(shape: [2, 3], elements: [7, 16, 27, 40, 55, 72])
        
        let c = Tensor(shape: [3, 1], elements: [7, 8, 9])
        let matmul = a.matmul(c) // Tensor(shape: [2, 1], elements: [50, 122])
        
        let zeros = Tensor(shape: [2, 3, 4])
        let ones = Tensor(shape: [2, 3, 4], element: 1)
        
        XCTAssertEqual(sum, Tensor(shape: [2, 3], elements: [8, 10, 12, 14, 16, 18]))
        XCTAssertEqual(mul, Tensor(shape: [2, 3], elements: [7, 16, 27, 40, 55, 72]))
        XCTAssertEqual(matmul, Tensor(shape: [2, 1], elements: [50, 122]))
        XCTAssertEqual(zeros, Tensor(shape: [2, 3, 4], elements: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        XCTAssertEqual(ones, Tensor(shape: [2, 3, 4], elements: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    }
    
    static var allTests : [(String, (TensorSwiftSample) -> () throws -> Void)] {
        return [
            ("testSample", testSample),
        ]
    }
}
