Pod::Spec.new do |s|
  s.name         = "TensorSwift"
  s.version      = "0.2.2"
  s.summary      = "TensorSwift is a lightweight library to calculate tensors, which has similar APIs to TensorFlow's."
  s.homepage     = "https://github.com/qoncept/TensorSwift"
  s.license      = { :type => "MIT", :file => "LICENSE" }
  s.authors      = { "Yuta Koshizawa" => "koshizawa@qoncept.co.jp", "Takehiro Araki" => "araki@qoncept.co.jp" }

  s.ios.deployment_target = "8.0"
  s.osx.deployment_target = "10.9"
  # s.watchos.deployment_target = "2.0"
  # s.tvos.deployment_target = "9.0"
  s.source       = { :git => "https://github.com/qoncept/TensorSwift.git", :tag => "#{s.version}" }
  s.source_files  = "Sources/*.swift"
end
