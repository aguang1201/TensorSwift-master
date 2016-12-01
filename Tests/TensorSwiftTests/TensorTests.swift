import XCTest
@testable import TensorSwift

class TensorTests: XCTestCase {
    func testIndex() {
        do {
            let a = Tensor(shape: [])
            XCTAssertEqual(a.index([]), 0)
        }
        
        do {
            let a = Tensor(shape: [7])
            XCTAssertEqual(a.index([3]), 3)
        }
        
        do {
            let a = Tensor(shape: [5, 7])
            XCTAssertEqual(a.index([1, 2]), 9)
        }
        
        do {
            let a = Tensor(shape: [5, 7, 11])
            XCTAssertEqual(a.index([3, 1, 2]), 244)
        }
    }
    
    func testAdd() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [7, 8, 9, 10, 11, 12])
            let r = a + b
            XCTAssertEqual(r, Tensor(shape: [2, 3], elements: [8, 10, 12, 14, 16, 18]))
        }
        
        do {
            let a = Tensor(shape: [2, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [2], elements: [100, 200])
            XCTAssertEqual(a + b, Tensor(shape: [2, 3, 2], elements: [101, 202, 103, 204, 105, 206, 107, 208, 109, 210, 111, 212]))
            XCTAssertEqual(b + a, Tensor(shape: [2, 3, 2], elements: [101, 202, 103, 204, 105, 206, 107, 208, 109, 210, 111, 212]))
        }
        
        do {
            let a = Tensor(shape: [2, 1, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [3, 2], elements: [100, 200, 300, 400, 500, 600])
            XCTAssertEqual(a + b, Tensor(shape: [2, 1, 3, 2], elements: [101, 202, 303, 404, 505, 606, 107, 208, 309, 410, 511, 612]))
            XCTAssertEqual(b + a, Tensor(shape: [2, 1, 3, 2], elements: [101, 202, 303, 404, 505, 606, 107, 208, 309, 410, 511, 612]))
        }
    }
    
    func testSub() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [12, 11, 10, 9, 8, 7])
            let r = a - b
            XCTAssertEqual(r, Tensor(shape: [2, 3], elements: [-11, -9, -7, -5, -3, -1]))
        }
        
        do {
            let a = Tensor(shape: [2, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [2], elements: [100, 200])
            XCTAssertEqual(a - b, Tensor(shape: [2, 3, 2], elements: [-99, -198, -97, -196, -95, -194, -93, -192, -91, -190, -89, -188]))
            XCTAssertEqual(b - a, Tensor(shape: [2, 3, 2], elements: [99, 198, 97, 196, 95, 194, 93, 192, 91, 190, 89, 188]))
        }
        
        do {
            let a = Tensor(shape: [2, 1, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [3, 2], elements: [100, 200, 300, 400, 500, 600])
            XCTAssertEqual(a - b, Tensor(shape: [2, 1, 3, 2], elements: [-99, -198, -297, -396, -495, -594, -93, -192, -291, -390, -489, -588]))
            XCTAssertEqual(b - a, Tensor(shape: [2, 1, 3, 2], elements: [99, 198, 297, 396, 495, 594, 93, 192, 291, 390, 489, 588]))
        }
    }
    
    func testMul() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [2, 3], elements: [7, 8, 9, 10, 11, 12])
            XCTAssertEqual(a * b, Tensor(shape: [2, 3], elements: [7, 16, 27, 40, 55, 72]))
            XCTAssertEqual(b * a, Tensor(shape: [2, 3], elements: [7, 16, 27, 40, 55, 72]))
        }
        
        do {
            let a = Tensor(shape: [2, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [2], elements: [10, 100])
            XCTAssertEqual(a * b, Tensor(shape: [2, 3, 2], elements: [10, 200, 30, 400, 50, 600, 70, 800, 90, 1000, 110, 1200]))
            XCTAssertEqual(b * a, Tensor(shape: [2, 3, 2], elements: [10, 200, 30, 400, 50, 600, 70, 800, 90, 1000, 110, 1200]))
        }
        
        do {
            let a = Tensor(shape: [2, 1, 3, 2], elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            let b = Tensor(shape: [3, 2], elements: [10, 100, 1000, -10, -100, -1000])
            XCTAssertEqual(a * b, Tensor(shape: [2, 1, 3, 2], elements: [10, 200, 3000, -40, -500, -6000, 70, 800, 9000, -100, -1100, -12000]))
            XCTAssertEqual(b * a, Tensor(shape: [2, 1, 3, 2], elements: [10, 200, 3000, -40, -500, -6000, 70, 800, 9000, -100, -1100, -12000]))
        }
        
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b: Float = 2.0
            XCTAssertEqual(a * b, Tensor(shape: [2, 3], elements: [2, 4, 6, 8, 10, 12]))
            XCTAssertEqual(b * a, Tensor(shape: [2, 3], elements: [2, 4, 6, 8, 10, 12]))
        }
    }
    
    func testDiv() {
        do {
            let a = Tensor(shape: [2, 3], elements: [2048, 512, 128, 32, 8, 2])
            let b = Tensor(shape: [2, 3], elements: [2, 4, 8, 16, 32, 64])
            XCTAssertEqual(a / b, Tensor(shape: [2, 3], elements: [1024, 128, 16, 2, 0.25, 0.03125]))
            XCTAssertEqual(b / a, Tensor(shape: [2, 3], elements: [0.0009765625, 0.0078125, 0.0625, 0.5, 4, 32]))
        }
        
        do {
            let a = Tensor(shape: [2, 3, 2], elements: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
            let b = Tensor(shape: [2], elements: [8, 2])
            XCTAssertEqual(a / b, Tensor(shape: [2, 3, 2], elements: [0.25, 2.0, 1.0, 8.0, 4.0, 32.0, 16.0, 128.0, 64.0, 512.0, 256.0, 2048.0]))
            XCTAssertEqual(b / a, Tensor(shape: [2, 3, 2], elements: [4.0, 0.5, 1.0, 0.125, 0.25, 0.03125, 0.0625, 0.0078125, 0.015625, 0.001953125, 0.00390625, 0.00048828125]))
        }
        
        do {
            let a = Tensor(shape: [3, 1, 2, 2], elements: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
            let b = Tensor(shape: [2, 2], elements: [8, 2, -8, -2])
            XCTAssertEqual(a / b, Tensor(shape: [3, 1, 2, 2], elements: [0.25, 2.0, -1.0, -8.0, 4.0, 32.0, -16.0, -128.0, 64.0, 512.0, -256.0, -2048.0]))
            XCTAssertEqual(b / a, Tensor(shape: [3, 1, 2, 2], elements: [4.0, 0.5, -1.0, -0.125, 0.25, 0.03125, -0.0625, -0.0078125, 0.015625, 0.001953125, -0.00390625, -0.00048828125]))
        }
        
        do {
            let a = Tensor(shape:[2, 3], elements: [ 1, 2, 4, 8, 16, 32])
            let b: Float = 2.0
            XCTAssertEqual(a / b, Tensor(shape: [2, 3], elements: [0.5, 1, 2, 4, 8, 16]))
            XCTAssertEqual(b / a, Tensor(shape: [2, 3], elements: [2, 1, 0.5, 0.25, 0.125, 0.0625]))
        }
    }
    
    func testMatmul() {
        do {
            let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
            let b = Tensor(shape: [3, 4], elements: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
            let r = a.matmul(b)
            XCTAssertEqual(r, Tensor(shape: [2, 4], elements: [74, 80, 86, 92, 173, 188, 203, 218]))
        }
        do {
            let a = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            let b = Tensor(shape: [3, 3], elements: [1, 1, 1, 2, 2, 2, 3, 3, 3])
            let r = a.matmul(b)
            XCTAssertEqual(r, Tensor(shape: [3, 3], elements: [6, 6, 6, 12, 12, 12, 18, 18, 18]))
        }
    }
    
    func testMatmulPerformance(){
        let a = Tensor(shape: [1000, 1000], element: 0.1)
        let b = Tensor(shape: [1000, 1000], element: 0.1)
        measure{
            _ = a.matmul(b)
        }
    }
    
    static var allTests : [(String, (TensorTests) -> () throws -> Void)] {
        return [
            ("testIndex", testIndex),
            ("testAdd", testAdd),
            ("testSub", testSub),
            ("testMul", testMul),
            ("testDiv", testDiv),
            ("testMatmul", testMatmul),
            ("testMatmulPerformance", testMatmulPerformance),
        ]
    }
}
