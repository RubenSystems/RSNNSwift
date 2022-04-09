import Foundation
import Accelerate


public struct MatrixSize : Equatable {
	
	public let rows: Int
	public let columns : Int
	
	public init (rows: Int, columns: Int) {
		self.rows = rows
		self.columns = columns
	}
}

public class Matrix {
	
	//Internal data
	private let matrixData: [Double]
	private let matrixSize: MatrixSize
	
	
	//Initialisers
	public init(data: [Double], size: MatrixSize) {
		self.matrixData = data
		self.matrixSize = size
	}
	
	public convenience init(_ data : [[Double]]) {
		self.init(data: data.flatMap {$0}, size: MatrixSize(rows: data.count, columns: data[0].count))
	}
	
	///Random initaliser double between 0 and -1 
	public static func random(size: MatrixSize, from: Double, to: Double) -> Matrix {
		let data = (0..<size.columns * size.rows).compactMap { _ in
			Double.random(in: from..<to)
		}
		return Matrix(data: data, size: size)
	}
	
	
	private enum MatrixOperationError: Error {
		case IncorrectDimensions
	}
}



//Getters
extension Matrix {
	///Get data returns the data as a 1d vector, useful for Accelerate operations
	public func data () -> [Double]{
		return self.matrixData
	}
	
	///Get size returns the size of the matrix.
	public func size() -> MatrixSize {
		return self.matrixSize
	}
	
	public func e() -> Matrix {
		let result = vForce.exp(self.data())
		return Matrix(data: result, size: self.size())
	}
	
	public func t() -> Matrix {
		var result : [Double] = Array<Double>(repeating: 0, count: self.size().rows * self.size().columns)
		
		vDSP_mtransD(self.data(), 1, &result, 1, vDSP_Length(self.size().columns), vDSP_Length(self.size().rows))
		
		return Matrix(data: result, size: MatrixSize(rows: self.size().columns, columns: self.size().rows))
	}
	
	public func abs() -> Matrix {
		var currentData = self.data()
		var result = [Double](repeating: 0, count: data().count)
		var n = Int32(data().count)
		vvfabs(&result, &currentData, &n)
		return Matrix(data: result, size: self.size())
	}
	
	public func formattedData() -> [[Double]] {
		return self.data().chunked(into: self.size().columns)
	}
	
	public func chunked(into: Int ) -> [Matrix] {
		let data = formattedData().chunked(into: into).compactMap {
			Matrix($0)
		}
		return data
	}
}


//Mathmetical operators
extension Matrix {
	public static func - (left : Matrix, right: Matrix) throws -> Matrix {
		guard left.size() == right.size() else {throw MatrixOperationError.IncorrectDimensions}
		
		let result = vDSP.subtract(left.data(), right.data())
		return Matrix(data: result, size: left.size())
	}
	
	///Defining addition between Matrix and Matrix
	public static func + (left: Matrix, right: Double) -> Matrix {
		let result = vDSP.add(right, left.data())
		return Matrix(data: result, size: left.size())
	}
	
	public static func + (left: Matrix, right: [Double]) -> Matrix {
		
		let result = left.formattedData().compactMap {
			vDSP.add($0, right)
		}.flatMap {$0}
		
		
		return Matrix(data: result, size: left.size())
	}
	
	///Defining addition between Matrix and Matrix
	public static func + (left: Matrix, right: Matrix) throws -> Matrix {
		guard left.size() == right.size() else {throw MatrixOperationError.IncorrectDimensions}
		
		let result = vDSP.add(left.data(), right.data())
		return Matrix(data: result, size: left.size())
	}
	
	///Defining addition
	public static func * (left: Matrix, right: Matrix) throws -> Matrix {
		guard left.size() == right.size() else {throw MatrixOperationError.IncorrectDimensions}
		
		let result = vDSP.multiply(left.data(), right.data())
		return Matrix(data: result, size: left.size())
	}
	
	public static func * (left: Matrix, right: Double) -> Matrix {
		
		let result = vDSP.multiply(right, left.data())
		return Matrix(data: result, size: left.size())
	}
	
	
	///Defining dot product for two matricies
	public static func dot(_ left: Matrix, _ right: Matrix) throws -> Matrix {
		guard left.size().columns == right.size().rows else {throw MatrixOperationError.IncorrectDimensions}
		
		var result : [Double] = Array(repeating: 0, count: left.size().rows * right.size().columns)
		
		vDSP_mmulD(left.data(), 1,
				   right.data(), 1,
				   &result, 1,
				   vDSP_Length(left.size().rows), vDSP_Length(right.size().columns), vDSP_Length(right.size().rows))
		
		return Matrix(data: result, size: MatrixSize(rows: left.size().rows, columns: right.size().columns))
	}
	
	
}


extension Array {
	public func chunked(into size: Int) -> [[Element]] {
		return stride(from: 0, to: count, by: size).map {
			Array(self[$0 ..< Swift.min($0 + size, count)])
		}
	}
}
