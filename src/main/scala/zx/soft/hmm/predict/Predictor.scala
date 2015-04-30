package zx.soft.hmm.algorithm

import java.util.Random

import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.{ DenseMatrix, DenseVector }

/**
  * 预测器，提供三种方法从HMM模型数据中预测结果，三种用户场景如下：
  * 1) 从一个给定的模型中生成一个输出状态序列（预测）
  * 2) 计算一个给定模型生成一个给定给定输出状态序列的最大似然值（模型最大似然）
  * 3) 对于一个给定的模型和观察序列，计算最可能的隐含序列（解码）
  */
object Predictor {

	/*
	 * 对于一个给定的模型和观察序列，预测出最可能的隐含序列；
	 * 如果使用log刻度来计算，设置scaled为true；
	 * 该过程需要高性能的硬件计算能力，但是对于大的观察序列数值计算更稳定些。
	 */
	def predict(model : Model, observedStates : Array[Int], scaled : Boolean) : Array[Int] = {
		Viterbi.run(model, observedStates, scaled)
	}

	/*
	 * 对于给定的HMM模型，预测逐步输出状态的序列
	 */
	def predict(model : Model, steps : Int) : Array[Int] = predict(model, steps, RandomUtils.getRandom())

	/*
	 * 对于给定的HMM模型，预测逐步输出状态的序列，含有参数seed
	 */
	def predict(model : Model, steps : Int, seed : Long) : Array[Int] = predict(model, steps, RandomUtils.getRandom(seed))

	/*
	 * 对于给定的HMM模型，使用给定的概率实验seed，预测逐步输出状态的序列
	 */
	private def predict(model : Model, steps : Int, rand : Random) : Array[Int] = {

		val cA = getCumulativeA(model)
		val cB = getCumulativeB(model)

		val cPi = getCumulativePi(model)

		val outputStates = new Array[Int](steps)

		/* 选择起始状态 */
		var hiddenState = 0

		var randnr = rand.nextDouble()
		while (cPi.get(hiddenState) < randnr) {
			hiddenState += 1
		}

		/*
		 * 根据累计概率分布画出逐步输出状态
		 */
		for (step <- 0 until steps) {
			/* 选择输出状态到给定的隐含状态 */
			randnr = rand.nextDouble()
			var outputState = 0

			while (cB.get(hiddenState, outputState) < randnr) {
				outputState += 1
			}

			outputStates(step) = outputState

			/* 选择下一个隐含状态 */
			randnr = rand.nextDouble();
			var nextHiddenState = 0

			while (cA.get(hiddenState, nextHiddenState) < randnr) {
				nextHiddenState += 1
			}

			hiddenState = nextHiddenState
		}

		outputStates

	}

	/*
	 * 对于给定的HMM模型，计算其累计传输概率矩阵；
	 * 每行i是隐含状态i的传输矩阵概率分布的累计分布。
	 */
	private def getCumulativeA(model : Model) : DenseMatrix = {

		val numHiddenStates = model.getNumHiddenStates()

		val A = model.getAMatrix
		val cA = new DenseMatrix(numHiddenStates, numHiddenStates)

		(0 until numHiddenStates).foreach(i => {

			var sum = 0.0
			(0 until numHiddenStates).foreach(j => {
				sum += A.get(i, j)
				cA.set(i, j, sum)
			})

			// 保证最后的一个隐含状态总是1.0的累计概率值
			cA.set(i, numHiddenStates - 1, 1.0)

		})

		cA

	}

	/*
	 * 对于给的的HMM模型，计算累计输出概率矩阵；
	 * 每行i是隐含状态i的输出概率分布的累计分布。
	 */
	private def getCumulativeB(model : Model) : DenseMatrix = {

		val numHiddenStates = model.getNumHiddenStates()
		val numOutputStates = model.getNumOutputStates()

		val B = model.getBMatrix
		val cB = new DenseMatrix(numHiddenStates, numOutputStates)

		(0 until numHiddenStates).foreach(i => {

			var sum = 0.0
			(0 until numOutputStates).foreach(j => {
				sum += B.get(i, j)
				cB.set(i, j, sum)
			})

			// 保证最后的一个输出状态总是1.0累计概率值
			cB.set(i, numOutputStates - 1, 1.0)

		})

		cB

	}

	/*
	 * 对于给定的HMM模型，计算起始隐含状态概率的累计分布
	 */
	private def getCumulativePi(model : Model) : DenseVector = {

		val numHiddenStates = model.getNumHiddenStates()

		val Pi = model.getPiVector
		val cPi = new DenseVector(Pi.size())

		var sum = 0.0
		(0 until numHiddenStates).foreach(i => {
			sum += Pi.get(i)
			cPi.set(i, sum)
		})

		// 保证最后一个隐含状态总是1.0累计概率值
		cPi.set(numHiddenStates - 1, 1.0)

		cPi

	}

	/*
	 * 返回给定模型产生的给定输出序列的最大似然值；
	 * 在方法内部，调用了前向算法来计算alpha值，然后使用重载函数来计算实际的模型似然值；
	 * 如果scaled设置true，那么使用log刻度来计算；
	 * 算法这种计算比较耗资源，但是对于大的输出序列数值稳定性较好。
	 */
	def modelLikelihood(model : Model, outputStates : Array[Int], scaled : Boolean) : Double = modelLikelihood(Forward.run(model, outputStates, scaled), scaled)

	def modelLikelihood(alpha : DenseMatrix, scaled : Boolean) : Double = {

		var likelihood = 0.0

		val numCols = alpha.numCols()
		val numRows = alpha.numRows()

		if (scaled) {
			(0 until numCols).foreach(i => likelihood += Math.exp(alpha.getQuick(numRows - 1, i)))

		} else {
			(0 until numCols).foreach(i => likelihood += alpha.getQuick(numRows - 1, i))

		}

		likelihood

	}

	/*
	 * 对于给定模型计算出来的输出序列，计算出该输出序列的似然值；
	 * 如果beta都是log刻度的，需要将scaled设置为true。
	 */
	def modelLikelihood(model : Model, outputSequence : Array[Int], beta : DenseMatrix, scaled : Boolean) : Double = {

		var likelihood = 0.0

		val B = model.getBMatrix
		val Pi = model.getPiVector

		val numHiddenStates = model.getNumHiddenStates()

		val firstOutput = outputSequence(0)
		if (scaled) {
			(0 until numHiddenStates).foreach(i =>
				likelihood += Pi.getQuick(i) * Math.exp(beta.getQuick(0, i)) * B.getQuick(i, firstOutput)
			)

		} else {
			(0 until numHiddenStates).foreach(i =>
				likelihood += Pi.getQuick(i) * beta.getQuick(0, i) * B.getQuick(i, firstOutput)
			)

		}

		likelihood

	}

}
