import UIKit

extension UIImage {
    /**
     *  重设图片大小
     */
    func reSizeImage(reSize:CGSize)->UIImage {
        //UIGraphicsBeginImageContext(reSize);
        UIGraphicsBeginImageContextWithOptions(reSize,false,UIScreen.main.scale);
        self.draw(in: CGRect(x:0, y:0, width:reSize.width, height:reSize.height));
        let reSizeImage:UIImage = UIGraphicsGetImageFromCurrentImageContext()!;
        let context = UIGraphicsGetCurrentContext()
        context?.setStrokeColor(UIColor(colorLiteralRed: 0.0, green: 0.0, blue: 0.0, alpha: 1.0).cgColor)
        UIGraphicsEndImageContext();
        return reSizeImage;
    }
    
    /**
     *  等比率缩放
     */
    func scaleImage(scaleSize:CGFloat)->UIImage {
        let reSize = CGSize(width: self.size.width * scaleSize, height:self.size.height * scaleSize)
        return reSizeImage(reSize: reSize)
    }
    
    func modifyImage()->UIImage{
        
        let reSize = CGSize(width: 320, height: 320)
        UIGraphicsBeginImageContext(reSize)
        
        self.draw(in: CGRect(x:0, y:0, width:reSize.width, height:reSize.height));
//        let context = UIGraphicsGetCurrentContext()
//        context?.setStrokeColor(UIColor(colorLiteralRed: 0.0, green: 0.0, blue: 0.0, alpha: 1.0).cgColor)
        
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return result!
    }
}
