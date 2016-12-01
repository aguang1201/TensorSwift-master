import UIKit
import TensorSwift

class ViewController: UIViewController {
    @IBOutlet private var canvasView: CanvasView!
    
    private let inputSize = 28
    private let classifier = Classifier(path: Bundle.main.resourcePath!)

    @IBAction func onPressClassifyButton(_ sender: UIButton) {
        let input: Tensor
        do {
            let image = canvasView.image
            
            let cgImage = image.cgImage!
            
            var pixels = [UInt8](repeating: 0, count: inputSize * inputSize)
            
            let context  = CGContext(data: &pixels, width: inputSize, height: inputSize, bitsPerComponent: 8, bytesPerRow: inputSize, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0)!
            context.clear(CGRect(x: 0.0, y: 0.0, width: CGFloat(inputSize), height: CGFloat(inputSize)))
            
            let rect = CGRect(x: 0.0, y: 0.0, width: CGFloat(inputSize), height: CGFloat(inputSize))
            context.draw(cgImage, in: rect)
            
            input = Tensor(shape: [Dimension(inputSize), Dimension(inputSize), 1], elements: pixels.map { -(Float($0) / 255.0 - 0.5) + 0.5 })
        }

        let estimatedLabel = classifier.classify(input)

        let alertController = UIAlertController(title: "\(estimatedLabel)", message: nil, preferredStyle: .alert)
        alertController.addAction(UIAlertAction(title: "Dismiss", style: .default) { _ in self.canvasView.clear() })
        present(alertController, animated: true, completion: nil)
    }
}

