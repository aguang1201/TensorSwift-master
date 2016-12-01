import XCTest
@testable import TensorSwift

class TensorNNTests: XCTestCase {
    func testMaxPool() {
        do {
            let a = Tensor(shape: [2,3,1], elements: [0,1,2,3,4,5])
            let r = a.maxPool(kernelSize: [1,3,1], strides: [1,1,1])
            XCTAssertEqual(r, Tensor(shape: [2,3,1], elements: [1,2,2,4,5,5]))
        }

        do {
            let a = Tensor(shape: [2,2,2], elements: [0,1,2,3,4,5,6,7])
            
            do {
                let r = a.maxPool(kernelSize:[1,2,1], strides: [1,1,1])
                XCTAssertEqual(r, Tensor(shape: [2,2,2], elements: [2, 3, 2, 3, 6, 7, 6, 7]))
            }
            
            do {
                let r = a.maxPool(kernelSize:[1,2,1], strides: [1,2,1])
                XCTAssertEqual(r, Tensor(shape: [2,1,2], elements: [2, 3, 6, 7]))
            }
        }
    }
    
    func testConv2d() {
        do {
            let a = Tensor(shape: [2,4,1], elements: [1,2,3,4,5,6,7,8])
            
            do {
                let filter = Tensor(shape: [2,1,1,2], elements: [1,2,1,2])
                let result = a.conv2d(filter: filter, strides: [1,1,1])
                XCTAssertEqual(result, Tensor(shape: [2,4,2], elements: [6,12,8,16,10,20,12,24,5,10,6,12,7,14,8,16]))
            }
            
            do {
                let filter = Tensor(shape: [1,1,1,5], elements: [1,2,1,2,3])
                let result = a.conv2d(filter: filter, strides: [1,1,1])
                XCTAssertEqual(result, Tensor(shape: [2,4,5], elements: [1,2,1,2,3,2,4,2,4,6,3,6,3,6,9,4,8,4,8,12,5,10,5,10,15,6,12,6,12,18,7,14,7,14,21,8,16,8,16,24]))
            }
        }
        
        do {
            let a = Tensor(shape: [2,2,4], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
            let filter = Tensor(shape: [1,1,4,2], elements: [1,2,1,2,3,2,1,1])
            let result = a.conv2d(filter: filter, strides: [1,1,1])
            XCTAssertEqual(result, Tensor(shape: [2,2,2], elements: [16, 16, 40, 44, 64, 72, 88, 100]))
        }
        
        do {
            let a = Tensor(shape: [4,2,2], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
            let filter = Tensor(shape: [2,2,2,1], elements: [1,2,1,2,3,2,1,1])
            let result = a.conv2d(filter: filter, strides: [2,2,1])
            XCTAssertEqual(result, Tensor(shape: [2,1,1], elements: [58,162]))
        }
        
        do {
            let a = Tensor(shape: [4,4,1], elements: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
            let filter = Tensor(shape: [3,3,1,1], elements: [1,2,1,2,3,2,1,1,1])
            let result = a.conv2d(filter: filter, strides: [3,3,1])
            XCTAssertEqual(result, Tensor(shape: [2,2,1], elements: [18,33,95,113]))
        }
        
        do {
            let a = Tensor(shape: [1,3,1], elements: [1,2,3])
            let filter = Tensor(shape: [1,3,1,2], elements: [1,1,2,2,3,3])
            let result = a.conv2d(filter: filter, strides: [1,1,1])
            XCTAssertEqual(result, Tensor(shape: [1,3,2], elements: [8, 8, 14, 14, 8, 8]))
        }
    }
    
    func testMaxPoolPerformance(){
        let image = Tensor(shape: [28,28,3], element: 0.1)
        measure{
            _ = image.maxPool(kernelSize: [2,2,1], strides: [2,2,1])
        }
    }
    
    func testConv2dPerformance(){
        let image = Tensor(shape: [28,28,1], element: 0.1)
        let filter = Tensor(shape: [5,5,1,16], element: 0.1)
        measure{
            _ = image.conv2d(filter: filter, strides: [1,1,1])
        }
    }
    
    static var allTests : [(String, (TensorNNTests) -> () throws -> Void)] {
        return [
            ("testMaxPool", testMaxPool),
            ("testConv2d", testConv2d),
            ("testMaxPoolPerformance", testMaxPoolPerformance),
            ("testConv2dPerformance", testConv2dPerformance),
        ]
    }
}
