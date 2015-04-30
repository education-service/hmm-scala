package zx.soft.hmm.algorithm

import java.util.Random

import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.{ DenseMatrix, DenseVector }

import scala.collection.mutable.HashMap

/**
  * HMM 数据模型
  */
class Model extends Cloneable {

	// 存储观察状态名称
	private var outputStateNames = HashMap.empty[String, Int]

	// 存储隐含状态名称
	private var hiddenStateNames = HashMap.empty[String, Int]

	/* 隐含状态数 */
	private var numHiddenStates : Int = 0

	/* 输出状态数 */
	private var numOutputStates : Int = 0

	/*
	 * 传输矩阵：表示隐含状态之间的传输概率，
	 * TransitionMatrix(i,j)表示从隐含状态i到隐含状态j的概率，
	 *  概率的一般表达式：P(h(t+1)=h_j | h(t) = h_i) = TransitionMatrix(i,j)，
	 *  因为需要确保每个隐含状态位于“左边”，所以可以使用下面的归一化条件来限制：
	 *      sum(TransitionMatrix(i,j),j=1..hiddenStates) = 1
	 */
	private var A : DenseMatrix = null

	/*
	 * 输出矩阵：表示观察一个特定输出状态得到一个隐含状态的概率，
	 * OutputMatrix(i,j)表示在隐含状态i的情况下，观察到输出状态j的概率，
	 * 概率的一般表达式：P(o(t)=o_j | h(t)=h_i) = outputMatrix(i,j)，
	 * 因为对于每个隐含状态都需要观察，所以可以使用下面的归一化条件来限制：
	 * sum(OutputMatrix(i,j),j=1..outputStates) = 1
	 */
	private var B : DenseMatrix = null

	/*
	 * 向量：表示初始隐含状态概率，
	 * 概率的一般表达式：P(h(0)=h_i) = initialProbabilities(i)，
	 * 需要通过下面的归一化条件来处理概率：
	 * sum(InitialProbabilities(i),i=1..hiddenStates) = 1
	 */
	private var Pi : DenseVector = null

	/**
	  * 克隆HMM数据模型
	  */
	override def clone() : Model = {

		val AMatrixClone = A.clone().asInstanceOf[DenseMatrix]
		val BMatrixClone = B.clone().asInstanceOf[DenseMatrix]

		val model = new Model(AMatrixClone, BMatrixClone, Pi.clone())
		if (hiddenStateNames != null) {
			model.hiddenStateNames = hiddenStateNames.clone()

		}
		if (outputStateNames != null) {
			model.outputStateNames = outputStateNames.clone()

		}

		model

	}

	/**
	  * 将另外一个HMM数据模型复制过来
	  */
	def assign(model : Model) {

		this.numHiddenStates = model.numHiddenStates;
		this.numOutputStates = model.numOutputStates;

		this.hiddenStateNames = model.hiddenStateNames;
		this.outputStateNames = model.outputStateNames;

		this.Pi = model.Pi.clone()

		this.A = model.A.clone().asInstanceOf[DenseMatrix]
		this.B = model.B.clone().asInstanceOf[DenseMatrix]

	}

	/**
	  * 通过给定的隐含状态数和输出状态数来构造一个有效的随机Hidden-Markov参数集，
	  * seed参数用于随机初始化，如果设置为0的话，表示使用当前时间。
	  */
	def this(numHiddenStates : Int, numOutputStates : Int, seed : Long) {
		this()

		this.numHiddenStates = numHiddenStates
		this.numOutputStates = numOutputStates

		this.A = new DenseMatrix(numHiddenStates, numHiddenStates)
		this.B = new DenseMatrix(numHiddenStates, numOutputStates)

		this.Pi = new DenseVector(numHiddenStates)

		initRandomParameters(seed)

	}

	/**
	  * 通过给定的隐含状态数和输出状态数构造一个有效的随机Hidden-Markov参数集。
	  */
	def this(numHiddenStates : Int, numOutputStates : Int) {
		this(numHiddenStates, numOutputStates, 0);
	}

	/**
	  * 通过制定的参数生成一个HMM
	  */
	def this(A : DenseMatrix, B : DenseMatrix, Pi : DenseVector) {
		this()

		this.numHiddenStates = Pi.size()
		this.numOutputStates = B.numCols()

		this.A = A
		this.B = B

		this.Pi = Pi

	}

	/**
	  * 初始化一个有效的随机HMM参数集合
	  */
	private def initRandomParameters(seed : Long) {

		/* 初始化随机数生成器 */
		val rand : Random = if (seed == 0) RandomUtils.getRandom() else RandomUtils.getRandom(seed)

		/* 初始化起始概率 */
		var sum : Double = 0
		(0 until numHiddenStates).foreach(i => {

			val nextRand = rand.nextDouble()
			Pi.set(i, nextRand)

			sum += nextRand

		})

		/* "归一化" 初始隐含状态概率向量来生成概率 */
		Pi = Pi.divide(sum).asInstanceOf[DenseVector]

		/* 初始化传输矩阵 */
		var values = Array.fill[Double](numHiddenStates)(0)
		(0 until numHiddenStates).foreach(i => {

			sum = 0
			(0 until numHiddenStates).foreach(j => {
				values(j) = rand.nextDouble()
				sum += values(j)
			})

			/* 归一化随机值获取概率 */
			(0 until numHiddenStates).foreach(j => values(j) /= sum)

			/* 设置传输矩阵的第i行的值 */
			A.set(i, values)

		})

		/* 初始化输出矩阵 */
		values = Array.fill[Double](numOutputStates)(0)
		(0 until numHiddenStates).foreach(i => {

			sum = 0
			(0 until numOutputStates).foreach(j => {
				values(j) = rand.nextDouble()
				sum += values(j)
			})

			/* 归一化随机值获取概率 */
			(0 until numOutputStates).foreach(j => values(j) /= sum)

			/* 设置输出矩阵的第i行的值 */
			B.set(i, values)

		})

	}

	def getNumHiddenStates() : Int = numHiddenStates

	def getNumOutputStates() : Int = numOutputStates

	def getAMatrix : DenseMatrix = A

	def getBMatrix : DenseMatrix = B

	def getPiVector : DenseVector = Pi

	def getHiddenStateNames() : Map[String, Int] = hiddenStateNames.toMap

	/**
	  * 注册一个隐含状态名称数组，假设位置i的隐含状态名称的ID为i
	  */
	def registerHiddenStateNames(stateNames : Array[String]) {

		if (stateNames != null) {
			(0 until stateNames.length).foreach(i => hiddenStateNames += stateNames(i) -> i)
		}

	}

	/**
	  * 注册一个隐含状态名称和状态ID的映射表
	  */
	def registerHiddenStateNames(stateNames : Map[String, Int]) {

		if (stateNames != null) {

			for (stateName <- stateNames) {
				hiddenStateNames += stateName
			}

		}

	}

	/**
	  * 通过隐含状态的ID查询其名称
	  */
	def getHiddenStateName(id : Int) : String = {

		if (hiddenStateNames.isEmpty) return null

		val names = hiddenStateNames.filter(e => e._2 == id).map(e => e._1).toSeq
		if (names.size > 0) names(0) else null

	}

	/**
	  * 通过隐含状态的名称查询其ID
	  */
	def getHiddenStateID(name : String) : Int = {

		hiddenStateNames.get(name) match {
			case None => -1
			case Some(id) => id
		}

	}

	def getOutputStateNames() : Map[String, Int] = outputStateNames.toMap

	/**
	  * 注册一个输出状态名称的数组，假设位置i的状态名称的ID为i
	  */
	def registerOutputStateNames(stateNames : Array[String]) {

		if (stateNames != null) {

			for (i <- 0 until stateNames.length) {
				outputStateNames += stateNames(i) -> i
			}

		}

	}

	/**
	  * 注册一个隐含状态名称和状态ID的映射表
	  */
	def registerOutputStateNames(stateNames : Map[String, Int]) {
		if (stateNames != null) {

			for (stateName <- stateNames) {
				outputStateNames += stateName
			}

		}
	}

	/**
	  * 通过输出状态的ID查询其名称
	  */
	def getOutputStateName(id : Int) : String = {

		if (outputStateNames.isEmpty) return null

		val names = outputStateNames.filter(e => e._2 == id).map(e => e._1).toSeq
		if (names.size > 0) names(0) else null

	}

	/**
	  * 通过输出状态的名称查询其ID
	  */
	def getOutputStateID(name : String) : Int = {

		outputStateNames.get(name) match {
			case None => -1
			case Some(id) => id

		}

	}

	/**
	  * 归一化HMM模型中的概率
	  */
	def normalize() {

		var isum = 0.0
		(0 until numHiddenStates).foreach(i => {

			isum += Pi.getQuick(i)

			var sum = 0.0
			(0 until numHiddenStates).foreach(j => sum += A.getQuick(i, j))

			if (sum != 1.0) {
				(0 until numHiddenStates).foreach(j => A.setQuick(i, j, A.getQuick(i, j) / sum))
			}

			sum = 0.0
			(0 until numOutputStates).foreach(j => sum += B.getQuick(i, j))

			if (sum != 1.0) {
				(0 until numOutputStates).foreach(j => B.setQuick(i, j, B.getQuick(i, j) / sum))
			}

		})

		if (isum != 1.0) {
			(0 until numHiddenStates).foreach(i => Pi.setQuick(i, Pi.getQuick(i) / isum))
		}
	}

}
