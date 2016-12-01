import XCTest

class DownloaderTests: XCTestCase {
    func testDownloadTestData() {
        let testData = downloadTestData()
        
        XCTAssertEqual(testData.images.count, 7840016)
        XCTAssertEqual(testData.images.sha1, "65e11ec1fd220343092a5070b58418b5c2644e26")
    }
}
