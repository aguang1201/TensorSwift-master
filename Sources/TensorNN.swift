import Darwin
#if os(iOS) || os(OSX)
import Accelerate
#endif

extension Tensor {
    public func softmax() -> Tensor {
        let exps = exp()
        let sum = exps.elements.reduce(0.0, +)
        return exps / sum
    }
    
    public func relu() -> Tensor {
        return Tensor(shape: shape, elements: elements.map { fmax($0, 0.0) })
    }
}

extension Tensor {
    public func maxPool(kernelSize: [Int], strides: [Int]) -> Tensor { // padding = Same
        precondition(shape.dimensions.count == 3, "`shape.dimensions.count` must be 3: \(shape.dimensions.count)")
        precondition(kernelSize.count == 3, "`ksize.count` must be 3: \(kernelSize.count)")
        precondition(kernelSize[2] == 1, "`ksize[3]` != 1 is not supported: \(kernelSize[2])")
        precondition(strides.count == 3, "`strides.count` must be 3: \(strides.count)")
        precondition(strides[2] == 1, "`strides[2]` != 1 is not supported: \(strides[2])")
        
        let inRows = shape.dimensions[0].value
        let inCols = shape.dimensions[1].value
        let numChannels = shape.dimensions[2].value
        
        let filterHeight = kernelSize[0]
        let filterWidth = kernelSize[1]
        
        let inMinDy = -(filterHeight - 1) / 2
        let inMaxDy = inMinDy + filterHeight - 1
        let inMinDx = -(filterWidth - 1) / 2
        let inMaxDx = inMinDx + filterWidth - 1
        
        let rowStride = strides[0]
        let colStride = strides[1]

        let outRows = shape.dimensions[0].value.ceilDiv(rowStride)
        let outCols = shape.dimensions[1].value.ceilDiv(colStride)
        
        // Initialize with -infinity for maximization.
        let elements = [Element](repeating: -Float.infinity, count: outCols * outRows * numChannels)
        
        for y in 0..<outRows {
            let inY0 = y * rowStride
            let inMinY = Swift.max(inY0 + inMinDy, 0)
            let inMaxY = Swift.min(inY0 + inMaxDy, inRows - 1)

            for inY in inMinY...inMaxY {
                var outPixelIndex = y * outCols
                for x in 0..<outCols {
                    let inX0 = x * colStride
                    let inMinX = Swift.max(inX0 + inMinDx, 0)
                    let inMaxX = Swift.min(inX0 + inMaxDx, inCols - 1)
                    
                    var inPointer = UnsafeMutablePointer<Element>(mutating: self.elements) + (inY * inCols + inMinX) * numChannels
                    for _ in inMinX...inMaxX {
                        var outPointer = UnsafeMutablePointer<Element>(mutating: elements) + outPixelIndex * numChannels
                        for _ in 0..<numChannels {
                            outPointer.pointee = Swift.max(outPointer.pointee, inPointer.pointee)
                            outPointer += 1
                            inPointer += 1
                        }
                    }
                    outPixelIndex += 1
                }
            }
        }
        
        return Tensor(shape: [Dimension(outRows), Dimension(outCols), Dimension(numChannels)], elements: elements)
    }
    
    
    public func conv2d(filter: Tensor, strides: [Int]) -> Tensor { // padding = Same
        let inChannels = filter.shape.dimensions[2].value
        
        precondition(shape.dimensions.count == 3, "`shape.dimensions.count` must be 3: \(shape.dimensions.count)")
        precondition(filter.shape.dimensions.count == 4, "`filter.shape.dimensions.count` must be 4: \(filter.shape.dimensions.count)")
        precondition(strides.count == 3, "`strides.count` must be 3: \(strides.count)")
        precondition(strides[2] == 1, "`strides[2]` must be 1")
        precondition(shape.dimensions[2].value == inChannels, "The number of channels of this tensor and the filter are not compatible: \(shape.dimensions[2]) != \(inChannels)")
        
        let inRows = shape.dimensions[0].value
        let inCols = shape.dimensions[1].value
        
        let filterHeight = filter.shape.dimensions[0].value
        let filterWidth = filter.shape.dimensions[1].value
        
        let inMinDy = -(filterHeight - 1) / 2
        let inMaxDy = inMinDy + filterHeight - 1
        let inMinDx = -(filterWidth - 1) / 2
        let inMaxDx = inMinDx + filterWidth - 1
        
        let rowStride = strides[0]
        let colStride = strides[1]
        
        let outRows = inRows.ceilDiv(rowStride)
        let outCols = inCols.ceilDiv(colStride)
        let outChannels = filter.shape.dimensions[3].value
        
        #if os(iOS) || os(OSX)
            let elementsPointer = UnsafePointer<Float>(elements)
            
            // a.shape == [outRows * outCols, rowSize]
            let rowSize = filterHeight * filterWidth * inChannels
            let a = [Float](repeating: 0, count: outRows * outCols * rowSize)
            for y in 0..<outRows {
                let inY0 = y * rowStride
                let inMinY = Swift.max(inY0 + inMinDy, 0)
                let inMaxY = Swift.min(inY0 + inMaxDy, inRows - 1)
                
                for x in 0..<outCols{
                    let inX0 = x * colStride
                    let inMinX = Swift.max(inX0 + inMinDx, 0)
                    let inMaxX = Swift.min(inX0 + inMaxDx, inCols - 1)
                    
                    // Add (x,y)'s patch as a vector
                    var dest = UnsafeMutablePointer<Float>(mutating: a) + ((y * outCols + x) * filterHeight - Swift.min(inY0 + inMinDy, 0)) * filterWidth * inChannels
                    var src = elementsPointer + (inMinY * inCols + inMinX) * inChannels
                    for _ in inMinY...inMaxY {
                        memcpy(dest - Swift.min(inX0 + inMinDx, 0) * inChannels, src, (inMinX...inMaxX).count * inChannels * MemoryLayout<Float>.size)
                        dest += filterWidth * inChannels
                        src += inCols * inChannels
                    }
                }
            }
            
            let result = Tensor(shape: [Dimension(outRows), Dimension(outCols), Dimension(outChannels)])
            
            let n = Int32(outChannels)
            let k = Int32(rowSize)
            
            // Calculate [M, N] matrix, it automatically turns into [outRows, outCols, outChannels] Tensor
            cblas_sgemm(
                CblasRowMajor,                                // Order
                CblasNoTrans,                                 // TransA
                CblasNoTrans,                                 // TransB
                Int32(outRows * outCols),                     // M
                n,                                            // N
                k,                                            // K
                1.0,                                          // alpha
                UnsafePointer<Float>(a),                      // A
                k,                                            // lda
                UnsafePointer<Float>(filter.elements),        // B
                n,                                            // ldb
                1.0,                                          // beta
                UnsafeMutablePointer<Float>(mutating: result.elements), // C
                n                                             // ldc
            )
            
            return result
        #else
            let elements = [Float](count: outCols * outRows * outChannels, repeatedValue: 0.0)
            
            for y in 0..<outRows {
                let inY0 = y * rowStride
                let inMinY = max(inY0 + inMinDy, 0)
                let inMaxY = min(inY0 + inMaxDy, inRows - 1)
                
                for x in 0..<outCols{
                    let inX0 = x * colStride
                    let inMinX = max(inX0 + inMinDx, 0)
                    let inMaxX = min(inX0 + inMaxDx, inCols - 1)
                    
                    let inYOffset = inY0 + inMinDy
                    let inXOffset = inX0 + inMinDx
                    
                    for inY in inMinY...inMaxY {
                        for inX in inMinX...inMaxX {
                            matmuladd(
                                inChannels, outChannels,
                                (inY * inCols + inX) * inChannels, UnsafeMutablePointer(self.elements),
                                ((inY - inYOffset) * filterWidth + (inX - inXOffset)) * inChannels * outChannels, UnsafeMutablePointer(filter.elements),
                                (y * outCols + x) * outChannels, UnsafeMutablePointer(elements)
                            )
                        }
                    }
                }
            }
            
            return Tensor(shape: [Dimension(outRows), Dimension(outCols), Dimension(outChannels)], elements: elements)
        #endif
    }
}

#if os(iOS) || os(OSX)
#else
    private func matmuladd(inCols1Rows2: Int, _ outCols: Int, _ o1: Int, _ vec: UnsafeMutablePointer<Float>, _ o2: Int, _ mat: UnsafeMutablePointer<Float>, _ oo: Int, _ out: UnsafeMutablePointer<Float>) {
        var v = vec + o1
        for i in 0..<inCols1Rows2 {
            var o = out + oo
            let left = v.memory
            for c in 0..<outCols {
                o.memory += left * mat[i * outCols + c + o2]
                o += 1
            }
            v += 1
        }
    }
#endif
