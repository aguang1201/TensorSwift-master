import Foundation

func downloadTestData() -> (images: Data, labels: Data) {
    let baseUrl = "http://yann.lecun.com/exdb/mnist/"
    
    var testImagesUrl = URL(string: baseUrl)!.appendingPathComponent("t10k-images-idx3-ubyte.gz")
    var testLabelsUrl = URL(string: baseUrl)!.appendingPathComponent("t10k-labels-idx1-ubyte.gz")
    
    print("download: \(testImagesUrl)")
    let testImages = try! Data(contentsOf: testImagesUrl)
    print("download: \(testLabelsUrl)")
    let testLabels = try! Data(contentsOf: testLabelsUrl)

    return (images: ungzip(testImages)!, labels: ungzip(testLabels)!)
}

private func ungzip(_ source: Data) -> Data? {
    guard source.count > 0 else {
        return nil
    }
    
    var stream: z_stream = z_stream.init(next_in: UnsafeMutablePointer<Bytef>(mutating: (source as NSData).bytes.bindMemory(to: Bytef.self, capacity: source.count)), avail_in: uint(source.count), total_in: 0, next_out: nil, avail_out: 0, total_out: 0, msg: nil, state: nil, zalloc: nil, zfree: nil, opaque: nil, data_type: 0, adler: 0, reserved: 0)
    guard inflateInit2_(&stream, MAX_WBITS + 32, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
        return nil
    }
    
    let data = NSMutableData()
    
    while stream.avail_out == 0 {
        let bufferSize = 0x10000
        let buffer: UnsafeMutablePointer<Bytef> = UnsafeMutablePointer<Bytef>.allocate(capacity: bufferSize)
        stream.next_out = buffer
        stream.avail_out = uint(MemoryLayout.size(ofValue: buffer))
        inflate(&stream, Z_FINISH)
        let length: size_t = MemoryLayout.size(ofValue: buffer) - Int(stream.avail_out)
        if length > 0 {
            data.append(buffer, length: length)
        }
        buffer.deallocate(capacity: bufferSize)
    }
    
    inflateEnd(&stream)
    return (NSData(data: data as Data) as Data)
}
