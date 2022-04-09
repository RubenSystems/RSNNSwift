import Foundation

public class Dataset: Codable {
	
	private let x_train : [[Double]]
	private let y_train : [[Double]]

	public var xtrain : Matrix {
		return Matrix(x_train)
	}
	
	public var ytrain : Matrix {
		return Matrix(y_train)
	}
	
}


extension Bundle {
	public func decode<T: Decodable>(_ type: T.Type, from file: String) -> T {
		guard let url = self.url(forResource: file, withExtension: nil) else {
			fatalError("Failed to locate \(file) in bundle.")
		}
		guard let data = try? Data(contentsOf: url) else {
			fatalError("Failed to load \(file) from bundle.")
		}
		let decoder = JSONDecoder()
		let loaded: T
		do {
			loaded = try decoder.decode(T.self, from: data)
			return loaded
		} catch (let error) {
			fatalError(error.localizedDescription)
		}
	}
}
