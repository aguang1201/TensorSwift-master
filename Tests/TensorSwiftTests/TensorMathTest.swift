import XCTest
@testable import TensorSwift

class TensorMathTest: XCTestCase {
    func testPow() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [6, 5, 4, 3, 2, 1])
            XCTAssertEqual(a ** b, Tensor(shape: [2, 3], elements: [1, 32, 81, 64, 25, 6]))
            XCTAssertEqual(b ** a, Tensor(shape: [2, 3], elements: [6, 25, 64, 81, 32, 1]))
        }
        
        do {
            let a = Tensor(shape: [2, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [2], elements: [2, 3])
            XCTAssertEqual(a ** b, Tensor(shape: [2, 3, 2], elements: [1, 8, 9, 64, 25, 216, 49, 512, 81, 1000, 121, 1728]))
            XCTAssertEqual(b ** a, Tensor(shape: [2, 3, 2], elements: [2, 9, 8, 81, 32, 729, 128, 6561, 512, 59049, 2048, 531441]))
        }
        
        do {
            let a = Tensor(shape: [2, 1, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [3, 2], elements: [6, 5, 4, 3, 2, 1])
            XCTAssertEqual(a ** b, Tensor(shape: [2, 1, 3, 2], elements: [1, 32, 81, 64, 25, 6, 117649, 32768, 6561, 1000, 121, 12]))
            XCTAssertEqual(b ** a, Tensor(shape: [2, 1, 3, 2], elements: [6, 25, 64, 81, 32, 1, 279936, 390625, 262144, 59049, 2048, 1]))
        }
        
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b: Float = 2.0
            XCTAssertEqual(a ** b, Tensor(shape: [2, 3], elements: [1, 4, 9, 16, 25, 36]))
            XCTAssertEqual(b ** a, Tensor(shape: [2, 3], elements: [2, 4, 8, 16, 32, 64]))
        }
    }
    
    func testSigmoid() {
        do {
            let a = Tensor(shape: [2, 3], elements: [-10, -1, 0, 1, 10, 100])
            XCTAssertEqual(a.sigmoid(), Tensor(shape: [2, 3], elements: [4.53978719e-05, 2.68941432e-01, 5.00000000e-01, 7.31058598e-01, 9.99954581e-01, 1.00000000e+00]))
        }
    }
}
