# RSNNSwift

This is a package to make basic neural networks in swift. 
This is an example of a neural netowrk: 

	NeuralNetwork(layers: [
		DenseLayer(inputSize: 784, outputSize: 128, activation: ReLU()),
		DenseLayer(inputSize: 128, outputSize: 10, activation: Sigmoid())
	])

Neural networks cannot be saved. However, layers can. An easy way to do this is: 

	let layers = [
		DenseLayer(inputSize: 784, outputSize: 128, activation: ReLU()),
		DenseLayer(inputSize: 128, outputSize: 10, activation: ReLU()),
		DenseLayer(inputSize: 10, outputSize: 128, activation: ReLU()),
		DenseLayer(inputSize: 128, outputSize: 784, activation: Sigmoid())
	]
	
	//Loading layers from files 
	for (k, v) in layers.enumerated(){
		v.load(filename: "\(k).json")
	}
	
	//Processing here
	
	//Saving layers to files
	for (k, v) in layers.enumerated(){
		v.save(filename: "\(k).json")
	}

To train a neural network, you call the fit function: 

	network.fit(
		x: <TRAINING DATA>,
		y: <TESTING DATA>,
		epochs: 10,
		learningRate: 0.1,
		batchSize: 64
	)
	
Data must be in the form of a matrix. To convert your data to a matrix, just call 

	Matrix(<ANY 2d ARRAY>)

or 
	Matrix(data: <ANY 1D Array>, size: <THE SHAPE>)



There are also activation functions. You can make your own 
activation functions by conforming your activation function to the 
protocol **ActivationFunction**. As an example, this is ReLU: 

	public struct ReLU : ActivationFunction {
		
		

		public init(){}

		public func run(matrix: Matrix) -> Matrix {

			let zeros : [Double] = (0..<matrix.data().count).map { _ in
				return 0
			}

			return Matrix(data: vDSP.maximum(matrix.data(), zeros), size: matrix.size())
		}
		
		public func der(matrix: Matrix) -> Matrix {
			return Matrix(data: matrix.data().map {
				$0 > 0 ? 1 : 0
			}, size: matrix.size())
		}

	}
	

