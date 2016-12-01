import XCTest
@testable import TensorSwiftTests

XCTMain([
     testCase(TensorSwiftTests.allTests),
     testCase(DimensionTests.allTests),
     testCase(PowerTests.allTests),
     testCase(TensorTests.allTests),
     testCase(TensorNNTests.allTests),
     testCase(CalculationPerformanceTests.allTests),
     testCase(TensorSwiftSample.allTests),
])
