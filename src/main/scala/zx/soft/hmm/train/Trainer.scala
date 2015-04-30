package zx.soft.hmm.algorithm

import java.util.{ Collection, Date, Iterator }

import org.apache.mahout.math.{ DenseMatrix, DenseVector }

import scala.util.control.Breaks._

/**
  * 模型训练，包含三个主要的算法：有监督学习、无监督学习和无监督Baum-Welch算法。
  */
object Trainer {

	/* 基于观察序列和隐含状态构建一个有监督的HMM模型的起始估计值 */
	def trainSupervised(numHiddenStates : Int, numOutputStates : Int, observedStates : Array[Int], hiddenStates : Array[Int], pseudoCount : Double) : Model = {

		/* 保证伪计数不为0 */
		val pseudo = if (pseudoCount == 0) Double.MinValue else pseudoCount

		val A = new DenseMatrix(numHiddenStates, numHiddenStates)
		val B = new DenseMatrix(numHiddenStates, numOutputStates)

		// 分配一个大于0的初始小概率，所以看不到的状态将不会获取一个0概率值
		A.assign(pseudo)
		B.assign(pseudo)

		// 考虑到没有先验知识，需要假设所有的初始隐含状态可能相等
		val Pi = new DenseVector(numHiddenStates)
		Pi.assign(1.0 / numHiddenStates)

		/* 循环序列计算出传输的数量 */
		countTransitions(A, B, observedStates, hiddenStates)

		/* 保证概率都被归一化 */
		(0 until numHiddenStates).foreach(i => {

			/* 计算传输矩阵当前行的概率和 */
			var sum = 0.0
			(0 until numHiddenStates).foreach(j => sum += A.getQuick(i, j))
			/* 归一化传输矩阵当前行 */
			(0 until numHiddenStates).foreach(j => A.setQuick(i, j, A.getQuick(i, j) / sum))

			/* 计算发射矩阵当前行的概率和 */
			sum = 0.0
			(0 until numOutputStates).foreach(j => sum += B.getQuick(i, j))
			/* 归一化发射矩阵当前行 */
			(0 until numOutputStates).foreach(j => B.setQuick(i, j, B.getQuick(i, j) / sum))

		})

		new Model(A, B, Pi)

	}

	/**
	  * 对于给定的观察序列和隐含序列，计算state->state和state->output传输的数量
	  */
	private def countTransitions(A : DenseMatrix, B : DenseMatrix, observedStates : Array[Int], hiddenStates : Array[Int]) {

		B.setQuick(hiddenStates(0), observedStates(0), B.getQuick(hiddenStates(0), observedStates(0)) + 1)

		(1 until observedStates.length).foreach(i => {

			A.setQuick(hiddenStates(i - 1), hiddenStates(i), A.getQuick(hiddenStates(i - 1), hiddenStates(i)) + 1)
			B.setQuick(hiddenStates(i), observedStates(i), B.getQuick(hiddenStates(i), observedStates(i)) + 1)

		})

	}

	/**
	  * 基于多个观察序列和隐含状态构建一个有监督的HMM模型起始估计值
	  */
	def trainSupervisedSequence(numHiddenStates : Int, numOutputStates : Int, observedSeq : Collection[Array[Int]], hiddenSeq : Collection[Array[Int]], pseudoCount : Double) : Model = {

		/* 保证伪计数不为0 */
		val pseudo = if (pseudoCount == 0) Double.MinValue else pseudoCount

		val A = new DenseMatrix(numHiddenStates, numHiddenStates)
		val B = new DenseMatrix(numHiddenStates, numOutputStates)

		val Pi = new DenseVector(numHiddenStates)

		/* 分配伪计数来避免0概率 */
		A.assign(pseudo)
		B.assign(pseudo)

		Pi.assign(pseudo)

		/* 循环训练计算传输数量 */
		val hiddenIter = hiddenSeq.iterator()
		val observedIter = observedSeq.iterator()

		while (hiddenIter.hasNext() && observedIter.hasNext()) {

			val hiddenStates = hiddenIter.next()
			val observedStates = observedIter.next()

			// 增加起始概率的数量
			Pi.setQuick(hiddenStates(0), Pi.getQuick(hiddenStates(0)) + 1)
			countTransitions(A, B, observedStates, hiddenStates)
		}

		/*保证所有概率归一化 */
		var isum = 0.0 // 起始概率的和
		(0 until numHiddenStates).foreach(i => {

			isum += Pi.getQuick(i)

			/* 计算传输矩阵当前行的概率和 */
			var sum = 0.0
			(0 until numHiddenStates).foreach(j => sum += A.getQuick(i, j))
			/* 归一化传输矩阵当前行 */
			(0 until numHiddenStates).foreach(j => A.setQuick(i, j, A.getQuick(i, j) / sum))

			/* 计算发射矩阵当前行的概率和 */
			sum = 0
			(0 until numOutputStates).foreach(j => sum += B.getQuick(i, j))
			/* 归一化发射矩阵当前行 */
			(0 until numOutputStates).foreach(j => B.setQuick(i, j, B.getQuick(i, j) / sum))

		})

		/* 归一化起始概率 */
		(0 until numHiddenStates).foreach(i => Pi.setQuick(i, Pi.getQuick(i) / isum))

		new Model(A, B, Pi)

	}

	/**
	  * 对于给定的与观察序列相关的起始模型，使用Viterbi训练算法迭代训练其参数
	  */
	def trainViterbi(initialModel : Model, observedStates : Array[Int], pseudoCount : Double, epsilon : Double = 0.0001, maxIterations : Int = 1000, scaled : Boolean = true) : Model = {

		/* 保证伪计数不为0 */
		val pseudo = if (pseudoCount == 0) Double.MinValue else pseudoCount

		val lastModel = initialModel.clone()
		val model = initialModel.clone()

		val viterbiPath = new Array[Int](observedStates.length)

		val phi = Array.fill[Int](observedStates.length - 1, initialModel.getNumHiddenStates())(0)
		val delta = Array.fill[Double](observedStates.length, initialModel.getNumHiddenStates())(0.0)

		/* 使用Viterbi训练算法迭代 */
		breakable {
			(0 until maxIterations).foreach(i => {

				/* 计算Viterbi路径 */
				Viterbi.run(viterbiPath, delta, phi, lastModel, observedStates, scaled)

				// Viterbi迭代使用viterbi路径更新概率
				val A = model.getAMatrix
				val B = model.getBMatrix

				/* 分配伪计数值 */
				A.assign(pseudo)
				B.assign(pseudo)

				/* 计算传输矩阵 */
				countTransitions(A, B, observedStates, viterbiPath)

				val numHiddenStates = model.getNumHiddenStates()
				val numOutputStates = model.getNumOutputStates()

				/* 归一化矩阵概率 */
				(0 until numHiddenStates).foreach(j => {

					var sum = 0.0
					/* 归一化传输的行 */
					(0 until numHiddenStates).foreach(k => sum += A.getQuick(j, k))
					(0 until numHiddenStates).foreach(k => A.setQuick(j, k, A.getQuick(j, k) / sum))

					/* 归一化发射矩阵的行 */
					sum = 0.0
					(0 until numOutputStates).foreach(k => sum += B.getQuick(j, k))
					(0 until numOutputStates).foreach(k => B.setQuick(j, k, B.getQuick(j, k) / sum))

				})

				/* 检查是否收敛 */
				if (checkConvergence(lastModel, model, epsilon)) {
					break;
				}
				/* 使用新的迭代模型数据覆盖上一次的迭代模型 */
				lastModel.assign(model);

			})
		}

		model

	}

	/**
	  * 构建一个随机的起始HMM，并使用BaumWelch算法迭代训练与给定观察训练相关的模型参数
	  */
	def trainBaumWelch(numHiddenStates : Int, numOutputStates : Int, observedStates : Array[Int], epsilon : Double = 0.0001, maxIterations : Int = 1000) : Model = {

		/* 构建随机的HMM */
		val model = new Model(numHiddenStates, numOutputStates, new Date().getTime())

		/* 通过给定的观察序列训练模型 */
		trainBaumWelch(model, observedStates, epsilon, maxIterations, true)

	}

	/*
	 * 对于给定的与观察训练相关的起始模型，使用Baum-Welch训练算法迭代训练参数。
	 * initialModel：需要迭代的起始模型
	 * observedSequence：观察状态序列
	 * epsilon：收敛值
	 * maxIterations：最大迭代次数
	 * scaled：使用log刻度实现前向/后向算法，这样的话计算资源消耗比较大，但是对于很长的输出序列可以提供更好的数值稳定性
	 */
	def trainBaumWelch(initialModel : Model, observedStates : Array[Int], epsilon : Double, maxIterations : Int, scaled : Boolean) : Model = {

		val model = initialModel.clone()
		val lastModel = initialModel.clone()

		val hiddenCount = model.getNumHiddenStates()
		val visibleCount = observedStates.length

		val alpha = new DenseMatrix(visibleCount, hiddenCount)
		val beta = new DenseMatrix(visibleCount, hiddenCount)

		/* 使用BaumWelch算法迭代训练 */
		breakable {
			for (it <- 0 until maxIterations) {

				val Pi = model.getPiVector

				val A = model.getAMatrix
				val B = model.getBMatrix

				val numHiddenStates = model.getNumHiddenStates()
				val numOutputStates = model.getNumOutputStates()

				/* 计算前向和后向因子 */
				Forward.run(model, alpha, observedStates, scaled)
				Backward.run(model, beta, observedStates, scaled)

				if (scaled) {
					logScaledBaumWelch(observedStates, model, alpha, beta)
				} else {
					unscaledBaumWelch(observedStates, model, alpha, beta)
				}

				// 归一化传输矩阵和发射矩阵的概率
				var isum = 0.0
				for (j <- 0 until numHiddenStates) {

					/* 归一化传输矩阵的行 */
					var sum = 0.0

					(0 until numHiddenStates).foreach(k => sum += A.getQuick(j, k))
					(0 until numHiddenStates).foreach(k => A.setQuick(j, k, A.getQuick(j, k) / sum))

					/* 归一化发射矩阵的行 */
					sum = 0.0

					(0 until numOutputStates).foreach(k => sum += B.getQuick(j, k))
					(0 until numOutputStates).foreach(k => B.setQuick(j, k, B.getQuick(j, k) / sum))

					/* 归一化起始概率的参数 */
					isum += Pi.getQuick(j)

				}

				/* 归一化起始概率 */
				(0 until numHiddenStates).foreach(i => Pi.setQuick(i, Pi.getQuick(i) / isum))

				/* 检查是否收敛 */
				if (checkConvergence(lastModel, model, epsilon)) {
					break
				}

				/* 使用新的迭代模型数据覆盖上一次的迭代模型 */
				lastModel.assign(model)

			}
		}

		return model

	}

	private def unscaledBaumWelch(observedStates : Array[Int], model : Model, alpha : DenseMatrix, beta : DenseMatrix) {

		val Pi = model.getPiVector

		val A = model.getAMatrix
		val B = model.getBMatrix

		val modelLikelihood = Predictor.modelLikelihood(alpha, false)

		val numHiddenStates = model.getNumHiddenStates()
		val numOutputStates = model.getNumOutputStates()

		val numObserv = observedStates.length

		for (i <- 0 until numHiddenStates) {
			Pi.setQuick(i, alpha.getQuick(0, i) * beta.getQuick(0, i))
		}

		/* 计算传输概率矩阵A */
		(0 until numHiddenStates).foreach(i => {
			(0 until numHiddenStates).foreach(j => {

				var temp = 0.0
				(0 until numObserv - 1).foreach(t => {
					temp += alpha.getQuick(t, i) * B.getQuick(j, observedStates(t + 1)) * beta.getQuick(t + 1, j)
				})

				A.setQuick(i, j, A.getQuick(i, j) * temp / modelLikelihood)

			})
		})

		/* 计算发射概率矩阵B */
		(0 until numHiddenStates).foreach(i => {
			(0 until numOutputStates).foreach(j => {

				var temp = 0.0
				(0 until numObserv).foreach(t => {
					// delta张量
					if (observedStates(t) == j) {
						temp += alpha.getQuick(t, i) * beta.getQuick(t, i)
					}
				})

				B.setQuick(i, j, temp / modelLikelihood)

			})

		})

	}

	private def logScaledBaumWelch(observedSequence : Array[Int], model : Model, alpha : DenseMatrix, beta : DenseMatrix) {

		val Pi = model.getPiVector

		val A = model.getAMatrix
		val B = model.getBMatrix

		val modelLikelihood = Predictor.modelLikelihood(alpha, true)

		val numHiddenStates = model.getNumHiddenStates()
		val numOutputStates = model.getNumOutputStates()

		val sequenceLen = observedSequence.length

		(0 until numHiddenStates).foreach(i => Pi.setQuick(i, Math.exp(alpha.getQuick(0, i) + beta.getQuick(0, i))))

		/* 计算传输概率矩阵 */
		(0 until numHiddenStates).foreach(i => {
			(0 until numHiddenStates).foreach(j => {

				var sum = Double.NegativeInfinity // log(0)
				(0 until sequenceLen - 1).foreach(t => {

					val temp = alpha.getQuick(t, i) + Math.log(B.getQuick(j, observedSequence(t + 1))) + beta.getQuick(t + 1, j)
					if (temp > Double.NegativeInfinity) {
						// 处理0概率值情况
						sum = temp + Math.log1p(Math.exp(sum - temp))
					}

				})

				A.setQuick(i, j, A.getQuick(i, j) * Math.exp(sum - modelLikelihood))

			})
		})

		/* 计算发射概率矩阵 */
		(0 until numHiddenStates).foreach(i => {
			(0 until numOutputStates).foreach(j => {

				var sum = Double.NegativeInfinity // log(0)

				(0 until sequenceLen).foreach(t => {
					// delta张量
					if (observedSequence(t) == j) {
						val temp = alpha.getQuick(t, i) + beta.getQuick(t, i)
						if (temp > Double.NegativeInfinity) {
							// 处理0概率值情况
							sum = temp + Math.log1p(Math.exp(sum - temp))
						}
					}
				})

				B.setQuick(i, j, Math.exp(sum - modelLikelihood))

			})

		})

	}

	/**
	  * 通过计算发射矩阵和传输矩阵的简单距离来检查两种HMM模型是否收敛
	  */
	def checkConvergence(oldModel : Model, newModel : Model, epsilon : Double) : Boolean = {

		/* 传输概率的收敛 */
		val oldA = oldModel.getAMatrix
		val newA = newModel.getAMatrix

		val numHiddenStates = oldModel.getNumHiddenStates()
		val numOutputStates = oldModel.getNumOutputStates()

		var diff = 0.0

		(0 until numHiddenStates).foreach(i => {
			(0 until numHiddenStates).foreach(j => {

				val tmp = oldA.getQuick(i, j) - newA.getQuick(i, j)
				diff += tmp * tmp

			})
		})

		var norm = Math.sqrt(diff)

		diff = 0.0

		/* 发射概率的收敛 */
		val oldB = oldModel.getBMatrix
		val newB = newModel.getBMatrix

		(0 until numHiddenStates).foreach(i => {
			(0 until numOutputStates).foreach(j => {

				val tmp = oldB.getQuick(i, j) - newB.getQuick(i, j)
				diff += tmp * tmp

			})
		})

		norm += Math.sqrt(diff)
		(norm < epsilon)

	}

}
