import Foundation
import Accelerate

public struct WeightBiasLoader: Codable {
	private let weights: [[Double]]
	private let bias: [[Double]]
	
	public var Weights : Matrix {
		return Matrix(weights)
	}
	
	public var Bias : [Double] {
		return bias[0]
	}
}


public class DenseLayer {
	public var a_output: Matrix?
	
	
	private let activationFunc : ActivationFunction
	public var weights : Matrix
	public var bias : [Double]
	public var output: Matrix?
	
	public var newWeights: Matrix?
	public var newBias: [Double]?
	
	
	public init(inputSize: Int, outputSize: Int , activation: ActivationFunction) {
		self.activationFunc = activation
		//Output size is the same as the number of neurons
		self.weights = Matrix.random(size: MatrixSize(rows: inputSize, columns : outputSize),from: -1, to: 1)
		
		self.bias = (0..<outputSize).map { _ in
			Double.random(in: -0.1...0.1)
		}
	}

	public func size() -> MatrixSize {
		return weights.size();
	}
	
	public func activation() -> ActivationFunction {
		return activationFunc
	}
	
	public func update() {
		self.weights = newWeights!
		newWeights = nil
		
		self.bias = newBias! 
		newBias = nil
	}
	
	public func load(filename: String) {
		let loader = Bundle.main.decode(WeightBiasLoader.self, from: filename)
		self.weights = loader.Weights
		self.bias = loader.Bias
	}
}

public class NeuralNetwork {
	private var layers: [DenseLayer]
	
	public init (layers: [DenseLayer]) {
		self.layers = layers
	}

	
	public func fit(x xFit: Matrix, y yFit: Matrix, epochs: Int, learningRate: Double, batchSize: Int = 64) {
		
		let chunkedX = xFit.formattedData().chunked(into: batchSize).compactMap { Matrix($0) }
		let chunkedY = yFit.formattedData().chunked(into: batchSize).compactMap { Matrix($0) }
		

		for epoch in 0...epochs {
			for (x, y) in zip(chunkedX, chunkedY){
				//FPass
				var prevOutput = x
				for i in 0..<self.layers.count {
					layers[i].output = ((try! Matrix.dot(prevOutput, layers[i].weights)) + layers[i].bias)
					layers[i].a_output = layers[i].activation().run(matrix: layers[i].output!)
					prevOutput = layers[i].a_output!
				}
				
				var error: Matrix?
				var partialDerivitave: Matrix
				//Backward pass
				for i in (0..<layers.count).reversed() {
					if i == layers.count - 1 {
						error = (try! layers[i].a_output! - y) * 2
						
					} else {
						error = try! Matrix.dot(error!, layers[i + 1].weights.t())
					}
					error = try! layers[i].activation().der(matrix: layers[i].a_output!) * error!
					partialDerivitave = try! Matrix.dot((i == 0 ? x : layers[i - 1].a_output!).t(), error!)
					layers[i].newWeights = try! layers[i].weights + (partialDerivitave * -learningRate)
					layers[i].newBias = layers[i].bias
					for c in partialDerivitave.formattedData() {
						layers[i].newBias = vDSP.subtract(layers[i].bias, vDSP.multiply(learningRate, c))
					}
				}
				_ = layers.map {$0.update()}
			}
			print(epoch)
		}
	}

	
	public func predict(x: Matrix) -> Matrix {
		var prevOutput = x
		for i in 0..<self.layers.count {
			let currentOutput = (try! Matrix.dot(prevOutput, layers[i].weights)) + layers[i].bias
			prevOutput = layers[i].activation().run(matrix: currentOutput)
		}
		return prevOutput
	}
}


extension Array where Element == Double {
	func average() -> Double {
		return reduce(0.0) {
			return $0 + $1/Double(abs( count))
		}
	}
	
}
