package zx.soft.hmm.algorithm

import org.apache.mahout.math.{ DenseMatrix, DenseVector }

/**
  * Viterbi算法，用于计算一个给定HMM模型和观察序列的最有可能的隐含序列
  */
object Viterbi {

	def run(model : Model, observations : Array[Int], scaled : Boolean) : Array[Int] = {

		val numStates = model.getNumHiddenStates()
		val numObserv = observations.length

		/* t时刻，最可能的隐含状态落在在状态i的概率 */
		val delta = Array.fill[Double](numObserv, numStates)(0.0)

		/* t时刻，前面的隐含状态变为状态i的最可能的状态 */
		val phi = Array.fill[Int](numObserv - 1, numStates)(0)

		val sequence = new Array[Int](numObserv)
		run(sequence, delta, phi, model, observations, scaled)

		sequence

	}

	def run(sequence : Array[Int], delta : Array[Array[Double]], phi : Array[Array[Int]], model : Model, observations : Array[Int], scaled : Boolean) {

		val Pi = model.getPiVector

		val A = model.getAMatrix
		val B = model.getBMatrix

		val numStates = model.getNumHiddenStates()
		val numObserv = observations.length

		if (scaled) {

			// 初始化
			(0 until numStates).foreach(i => delta(0)(i) = Math.log(Pi.getQuick(i) * B.getQuick(i, observations(0))))

			// 根据时间迭代
			(1 until numObserv).foreach(t => {

				// 在隐含状态上迭代
				(0 until numStates).foreach(i => {

					// 计算出达到该状态的最大概率和最可能的状态
					var maxState = 0
					var maxProb = delta(t - 1)(0) + Math.log(A.getQuick(0, i))

					(1 until numStates).foreach(j => {
						val prob = delta(t - 1)(j) + Math.log(A.getQuick(j, i))
						if (prob > maxProb) {
							maxProb = prob
							maxState = j
						}
					})

					delta(t)(i) = maxProb + Math.log(B.getQuick(i, observations(t)))
					phi(t - 1)(i) = maxState

				})

			})

		} else {

			// 初始化
			(0 until numStates).foreach(i => delta(0)(i) = Pi.getQuick(i) * B.getQuick(i, observations(0)))

			// 根据时间迭代
			(1 until numObserv).foreach(t => {

				// 在隐含状态上迭代
				(0 until numStates).foreach(i => {

					// 计算出达到该状态的最大概率和最可能的状态
					var maxState = 0
					var maxProb = delta(t - 1)(0) * A.getQuick(0, i)

					(1 until numStates).foreach(j => {
						val prob = delta(t - 1)(j) * A.getQuick(j, i)
						if (prob > maxProb) {
							maxProb = prob
							maxState = j
						}
					})

					delta(t)(i) = maxProb * B.getQuick(i, observations(t))
					phi(t - 1)(i) = maxState

				})

			})

		}

		/*
		 * 对于初始情况，计算出最可能的最终状态
		 */
		var maxProb = if (scaled) Double.NegativeInfinity else 0.0

		(0 until numStates).foreach(i => {
			if (delta(observations.length - 1)(i) > maxProb) {

				maxProb = delta(observations.length - 1)(i)
				sequence(observations.length - 1) = i

			}

		})

		// 回溯，找出最可能的隐含序列
		for (t <- observations.length - 2 to 0 by -1) {
			sequence(t) = phi(t)(sequence(t + 1))
		}

	}

}
