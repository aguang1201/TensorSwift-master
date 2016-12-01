import XCTest
import TensorSwift
@testable import MNIST

class ClassifierTests: XCTestCase {
    func testClassify() {
        
        let classifier = Classifier(path: Bundle(for: ViewController.self).resourcePath!)
        let (images, labels) = downloadTestData()
        
        let count = 1000
        
        let xArray: [[Float]] = images.withUnsafeBytes { ptr in
            [UInt8](UnsafeBufferPointer(start: UnsafePointer<UInt8>(ptr + 16), count: 28 * 28 * count))
                .map { Float($0) / 255.0 }
                .grouped(28 * 28)
        }
        
        let yArray: [Int] = labels.withUnsafeBytes { ptr in
            [UInt8](UnsafeBufferPointer(start: UnsafePointer<UInt8>(ptr + 8), count: count))
                .map { Int($0) }
        }
        
        let accuracy = Float(zip(xArray, yArray)
            .reduce(0) { $0 + (classifier.classify(Tensor(shape: [28, 28, 1], elements: $1.0)) == $1.1 ? 1 : 0) })
            / Float(yArray.count)
        
        print("accuracy: \(accuracy)")
        
        XCTAssertGreaterThan(accuracy, 0.97)
    }
}
