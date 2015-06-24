package pipelines.jonas

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import breeze.linalg._
import evaluation.MulticlassClassifierEvaluator
import loaders.{CsvDataLoader, LabeledData, ImageNetLoader}
import nodes.learning.{BlockLinearMapper, BlockLeastSquaresEstimator}
import nodes.stats.{LinearRectifier, PaddedFFT, RandomSignNode}
import nodes.util.{ZipVectors, ClassLabelIndicatorsFromIntLabels, MaxClassifier}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pipelines._
import scopt.OptionParser


object MnistRandomFFT extends Serializable with Logging {
  val appName = "JonasMnistRandomFFT"

  def run(sc: SparkContext, conf: MnistRandomFFTConfig) {
    // This is a property of the MNIST Dataset (digits 0 - 9)
    val numClasses = 10

    val randomSignSource = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))

    // The number of pixels in an MNIST image (28 x 28 = 784)
    val mnistImageSize = 784

    // Because the mnistImageSize is 784, we get 512 PaddedFFT features per FFT.
    // So, calculate how many FFTs are needed per block to get the desired block size.
    val fftsPerBatch = conf.blockSize / 512
    val numFFTBatches = math.ceil(conf.numFFTs.toDouble/fftsPerBatch).toInt

    val startTime = System.nanoTime()

    // val train = LabeledData(
    //   CsvDataLoader(sc, conf.trainLocation, conf.numPartitions)
    //     // The pipeline expects 0-indexed class labels, but the labels in the file are 1-indexed
    //     .map(x => (x(0).toInt - 1, x(1 until x.length)))
    //     .cache())
    val train = 
        ImageNetLoader(sc, conf.trainLocation, conf.trainLocationLabels)
        .cache()

    val labels = ClassLabelIndicatorsFromIntLabels(numClasses).apply(train.map(_.label))

    val batchFeaturizer = (0 until numFFTBatches).map { batch =>
      (0 until fftsPerBatch).map { x =>
        RandomSignNode(mnistImageSize, randomSignSource) then PaddedFFT then LinearRectifier(0.0)
      }
    }

    val data_dense_vector = train.map(_.image.getSingleChannelAsFloatArray())
    val dd2 = data_dense_vector.map(convert(_, Double))
    val data_dense_vector_breeze = dd2.map(breeze.linalg.DenseVector[Double](_))

    val trainingBatches = batchFeaturizer.map { x =>
      ZipVectors(x.map(y => y.apply(data_dense_vector_breeze))).cache()
    }

    // Train the model
    val blockLinearMapper = new BlockLeastSquaresEstimator(
      conf.blockSize, 1, conf.lambda.getOrElse(0)).fit(trainingBatches, labels)

    val test =
      ImageNetLoader(sc, conf.testLocation, conf.testLocationLabels)
        .cache()

    val actual = test.map(_.label)
    val data_dense_vector_test = test.map(_.image.getSingleChannelAsFloatArray())
    val dd2_test = data_dense_vector.map(convert(_, Double))
    val data_dense_vector_breeze_test = dd2_test.map(breeze.linalg.DenseVector[Double](_))

    val testBatches = batchFeaturizer.map { x =>
      ZipVectors(x.map(y => y.apply(data_dense_vector_breeze_test))).cache()
    }

    // Calculate train error
    blockLinearMapper.applyAndEvaluate(trainingBatches,
      (trainPredictedValues: RDD[DenseVector[Double]]) => {
        val predicted = MaxClassifier(trainPredictedValues)
        val evaluator = MulticlassClassifierEvaluator(predicted, train.map(_.label), numClasses)
        logInfo("Train Error is " + (100 * evaluator.totalError) + "%")
      }
    )

    // Calculate test error
    blockLinearMapper.applyAndEvaluate(testBatches,
      (testPredictedValues: RDD[DenseVector[Double]]) => {
        val predicted = MaxClassifier(testPredictedValues)
        val evaluator = MulticlassClassifierEvaluator(predicted, actual, numClasses)
        logInfo("TEST Error is " + (100 * evaluator.totalError) + "%")
      }
    )

    val endTime = System.nanoTime()
    logInfo(s"Pipeline took ${(endTime - startTime)/1e9} s")
  }

  case class MnistRandomFFTConfig(
      trainLocation: String = "",
      trainLocationLabels: String = "",
      testLocation: String = "",
      testLocationLabels: String = "",
      numFFTs: Int = 200,
      blockSize: Int = 2048,
      numPartitions: Int = 10,
      lambda: Option[Double] = None,
      seed: Long = 0)

  def parse(args: Array[String]): MnistRandomFFTConfig = new OptionParser[MnistRandomFFTConfig](appName) {
    head(appName, "0.1")
    help("help") text("prints this usage text")
    opt[String]("trainLocation") required() action { (x,c) => c.copy(trainLocation=x) }
    opt[String]("trainLocationLabels") required() action { (x,c) => c.copy(trainLocationLabels=x) }
    opt[String]("testLocation") required() action { (x,c) => c.copy(testLocation=x) }
    opt[String]("testLocationLabels") required() action { (x,c) => c.copy(testLocationLabels=x) }
    opt[Int]("numFFTs") action { (x,c) => c.copy(numFFTs=x) }
    opt[Int]("blockSize") validate { x =>
      // Bitwise trick to test if x is a power of 2
      if (x % 512 == 0) {
        success
      } else  {
        failure("Option --blockSize must be divisible by 512")
      }
    } action { (x,c) => c.copy(blockSize=x) }
    opt[Int]("numPartitions") action { (x,c) => c.copy(numPartitions=x) }
    opt[Double]("lambda") action { (x,c) => c.copy(lambda=Some(x)) }
    opt[Long]("seed") action { (x,c) => c.copy(seed=x) }
  }.parse(args, MnistRandomFFTConfig()).get

  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {
    val appConfig = parse(args)

    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]")
    val sc = new SparkContext(conf)
    run(sc, appConfig)

    sc.stop()
  }
}
