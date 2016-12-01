import XCTest
@testable import TensorSwift

class PowerTests: XCTestCase {

    func testScalar() {
        let tensor = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
        let scalar = Tensor.Element(2)
        
        XCTAssertEqual(tensor ** scalar, Tensor(shape: [2, 2], elements: [1, 4, 9, 16]))
        XCTAssertEqual(scalar ** tensor, Tensor(shape: [2, 2], elements: [2, 4, 8, 16]))
        
        XCTAssertEqual(scalar ** tensor ** scalar, Tensor(shape: [2, 2], elements: [2, 16, 512, 65536]))
    }
    
    func testMatrices() {
        let tensor = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
        let tensor2 = Tensor(shape: [2, 2], elements: [2, 2, 2, 2])
        
        XCTAssertEqual(tensor ** tensor2, Tensor(shape: [2, 2], elements: [1, 4, 9, 16]))
        XCTAssertEqual(tensor ** tensor, Tensor(shape: [2, 2], elements: [1, 4, 27, 256]))
        
        XCTAssertEqual(tensor ** tensor2 ** tensor, Tensor(shape: [2, 2], elements: [1, 16, powf(3, 8), powf(4, 16)]))
    }

    static var allTests : [(String, (PowerTests) -> () throws -> Void)] {
        return [
            ("testScalar", testScalar),
            ("testMatrices", testMatrices),
        ]
    }
}
