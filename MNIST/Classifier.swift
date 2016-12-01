import Foundation
import TensorSwift

public struct Classifier {
    public let W_conv1: Tensor
    public let b_conv1: Tensor
    public let W_conv2: Tensor
    public let b_conv2: Tensor
    public let W_fc1: Tensor
    public let b_fc1: Tensor
    public let W_fc2: Tensor
    public let b_fc2: Tensor
    
    public func classify(_ x_image: Tensor) -> Int {
        let h_conv1 = (x_image.conv2d(filter: W_conv1, strides: [1, 1, 1]) + b_conv1).relu()
        let h_pool1 = h_conv1.maxPool(kernelSize: [2, 2, 1], strides: [2, 2, 1])
        
        let h_conv2 = (h_pool1.conv2d(filter: W_conv2, strides: [1, 1, 1]) + b_conv2).relu()
        let h_pool2 = h_conv2.maxPool(kernelSize: [2, 2, 1], strides: [2, 2, 1])
        
        let h_pool2_flat = h_pool2.reshaped([1, Dimension(7 * 7 * 64)])
        let h_fc1 = (h_pool2_flat.matmul(W_fc1) + b_fc1).relu()
        
        let y_conv = (h_fc1.matmul(W_fc2) + b_fc2).softmax()

        return y_conv.elements.enumerated().max { $0.1 < $1.1 }!.0
    }
}

extension Classifier {
    public init(path: String) {
        W_conv1 = Tensor(shape: [5, 5, 1, 32], elements: loadFloatArray(path, file: "W_conv1"))
        b_conv1 = Tensor(shape: [32], elements: loadFloatArray(path, file: "b_conv1"))
        W_conv2 = Tensor(shape: [5, 5, 32, 64], elements: loadFloatArray(path, file: "W_conv2"))
        b_conv2 = Tensor(shape: [64], elements: loadFloatArray(path, file: "b_conv2"))
        W_fc1 = Tensor(shape: [Dimension(7 * 7 * 64), 1024], elements: loadFloatArray(path, file: "W_fc1"))
        b_fc1 = Tensor(shape: [1024], elements: loadFloatArray(path, file: "b_fc1"))
        W_fc2 = Tensor(shape: [1024, 10], elements: loadFloatArray(path, file: "W_fc2"))
        b_fc2 = Tensor(shape: [10], elements: loadFloatArray(path, file: "b_fc2"))
    }
}

private func loadFloatArray(_ directory: String, file: String) -> [Float] {
    let data = try! Data(contentsOf: URL(fileURLWithPath: directory.stringByAppendingPathComponent(file)))
    return Array(UnsafeBufferPointer(start: UnsafeMutablePointer<Float>(mutating: (data as NSData).bytes.bindMemory(to: Float.self, capacity: data.count)), count: data.count / 4))
}
