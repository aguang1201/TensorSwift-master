# TensorSwift

_TensorSwift_ is a lightweight library to calculate tensors, which has similar APIs to [_TensorFlow_](https://www.tensorflow.org/)'s. _TensorSwift_ is useful to simulate calculating tensors in Swift __using models trained by _TensorFlow___.

```swift
let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
let b = Tensor(shape: [2, 3], elements: [7, 8, 9, 10, 11, 12])
let sum = a + b // Tensor(shape: [2, 3], elements: [8, 10, 12, 14, 16, 18])
let mul = a * b // Tensor(shape: [2, 3], elements: [7, 16, 27, 40, 55, 72])

let c = Tensor(shape: [3, 1], elements: [7, 8, 9])
let matmul = a.matmul(c) // Tensor(shape: [2, 1], elements: [50, 122])

let zeros = Tensor(shape: [2, 3, 4])
let ones = Tensor(shape: [2, 3, 4], element: 1)
```

## Deep MNIST for Experts

![deep-mnist.gif](Resources/DeepMnist.gif)

The following code shows how to simulate [Deep MNIST for Experts](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html), a tutorial of _TensorFlow_, by _TensorSwift_.

```swift
public struct Classifier {
    public let W_conv1: Tensor
    public let b_conv1: Tensor
    public let W_conv2: Tensor
    public let b_conv2: Tensor
    public let W_fc1: Tensor
    public let b_fc1: Tensor
    public let W_fc2: Tensor
    public let b_fc2: Tensor

    public func classify(x_image: Tensor) -> Int {
        let h_conv1 = (x_image.conv2d(filter: W_conv1, strides: [1, 1, 1]) + b_conv1).relu()
        let h_pool1 = h_conv1.maxPool(kernelSize: [2, 2, 1], strides: [2, 2, 1])
        
        let h_conv2 = (h_pool1.conv2d(filter: W_conv2, strides: [1, 1, 1]) + b_conv2).relu()
        let h_pool2 = h_conv2.maxPool(kernelSize: [2, 2, 1], strides: [2, 2, 1])
        
        let h_pool2_flat = h_pool2.reshaped([1, 7 * 7 * 64])
        let h_fc1 = (h_pool2_flat.matmul(W_fc1) + b_fc1).relu()
        
        let y_conv = (h_fc1.matmul(W_fc2) + b_fc2).softmax()

        return y_conv.elements.enumerated().max { $0.1 < $1.1 }!.0
    }
}
```

## Installation

### Swift Package Manager

```swift
.Package(url: "git@github.com:qoncept/TensorSwift.git", majorVersion: 0, minor: 2),
```

### CocoaPods

```
pod 'TensorSwift', '~> 0.2'
```

### Carthage

```
github "qoncept/TensorSwift" ~> 0.2
```

## License

[The MIT License](LICENSE)

