package zx.soft.hmm.algorithm

import org.apache.mahout.math.{ DenseMatrix, DenseVector }

/**
 * Implementation of the Backward algorithm
 */
object AlgoBackward {

	def run(model : Model, observations : Array[Int], scaled : Boolean) : DenseMatrix = {

		val beta = new DenseMatrix(observations.length, model.getNumHiddenStates())
		run(model, beta, observations, scaled)

		beta

	}

	def run(model : Model, beta : DenseMatrix, observations : Array[Int], scaled : Boolean) {

		val A = model.getAMatrix
		val B = model.getBMatrix

		val numStates = model.getNumHiddenStates()
		val numObserv = observations.length

		if (scaled) {

			/*
       * Initialization
       */
			(0 until numStates).foreach(i => beta.setQuick(numObserv - 1, i, 0))

			for (t <- numObserv - 2 to 0 by -1) {
				(0 until numStates).foreach(i => {

					var sum = Double.NegativeInfinity // log(0)
					(0 until numStates).foreach(j => {

						val tmp = beta.getQuick(t + 1, j) + Math.log(A.getQuick(i, j)) + Math.log(B.getQuick(j, observations(t + 1)))
						if (tmp > Double.NegativeInfinity) {
							// handle log(0)
							sum = tmp + Math.log1p(Math.exp(sum - tmp))
						}
					})

					beta.setQuick(t, i, sum)

				})
			}

		} else {
			/*
       * Initialization
       */
			(0 until numStates).foreach(i => beta.setQuick(numObserv - 1, i, 1))

			for (t <- numObserv - 2 to 0 by -1) {
				(0 until numStates).foreach(i => {

					var sum = 0.0
					(0 until numStates).foreach(j => sum += beta.getQuick(t + 1, j) * A.getQuick(i, j) * B.getQuick(j, observations(t + 1)))
					beta.setQuick(t, i, sum)

				})

			}

		}

	}

}
