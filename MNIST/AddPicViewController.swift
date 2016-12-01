//
//  AddPicViewController.swift
//  TensorSwift
//
//  Created by Mrzhou on 2016/11/28.
//
//

import UIKit
import TensorSwift

class AddPicViewController: UIViewController,UIImagePickerControllerDelegate,UINavigationControllerDelegate {

    private let inputSize = 28
    private let classifier = Classifier(path: Bundle.main.resourcePath!)
    
    @IBOutlet weak var picImageView: UIImageView!
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        
    }

    @IBAction func loadPicture(_ sender: UIButton) {
        if UIImagePickerController.isSourceTypeAvailable(.photoLibrary) {
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.allowsEditing = false
            imagePicker.sourceType = .photoLibrary
            
            self.present(imagePicker, animated: true, completion: nil)
        }
    }
    @IBAction func learningAction(_ sender: UIButton) {
        let input: Tensor
        do {
//            let reSize = CGSize(width: 320, height: 320)
//            let image = picImageView.image?.reSizeImage(reSize: reSize)
            let image = picImageView.image?.modifyImage()
            
            let cgImage = image?.cgImage!
            
            var pixels = [UInt8](repeating: 0, count: inputSize * inputSize)
            
            let context  = CGContext(data: &pixels, width: inputSize, height: inputSize, bitsPerComponent: 8, bytesPerRow: inputSize, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0)!
            context.clear(CGRect(x: 0.0, y: 0.0, width: CGFloat(inputSize), height: CGFloat(inputSize)))
            
            let rect = CGRect(x: 0.0, y: 0.0, width: CGFloat(inputSize), height: CGFloat(inputSize))
            context.draw(cgImage!, in: rect)
            
            input = Tensor(shape: [Dimension(inputSize), Dimension(inputSize), 1], elements: pixels.map { -(Float($0) / 255.0 - 0.5) + 0.5 })
        }
        
        let estimatedLabel = classifier.classify(input)
        
        let alertController = UIAlertController(title: "\(estimatedLabel)", message: nil, preferredStyle: .alert)
        alertController.addAction(UIAlertAction(title: "Dismiss", style: .default))
        present(alertController, animated: true, completion: nil)
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picImageView.image = info[UIImagePickerControllerOriginalImage] as? UIImage
        picImageView.contentMode = .scaleAspectFill
        picImageView.clipsToBounds = true
        
        dismiss(animated: true, completion: nil)
    }

    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destinationViewController.
        // Pass the selected object to the new view controller.
    }
    */

}
