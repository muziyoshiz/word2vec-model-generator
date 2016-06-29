package jp.muziyoshiz.word2vec

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}

object Word2VecModelGenerator {

  def main(args: Array[String]) {
    require (args.length == 5, "argument: <textFilePath> <modelPath> <numPartition> <minCount> <vectorSize>")

    val textFilePath = args(0)
    val modelPath = args(1)
    val numPartition = args(2).toInt
    val minCount = args(3).toInt
    val vectorSize = args(4).toInt

    val startTime = System.nanoTime

    val conf = new SparkConf().setAppName("Word2VecModelGenerator")
    val sc = new SparkContext(conf)

    try {
      val input = sc.textFile(textFilePath).repartition(sc.defaultParallelism * 3).map(line => line.split("""[ ,\."'\?!@\[\]{}:;()\|#=^~\\+*<>/_]+""").filter(_.length > 1).toSeq)

      val word2vec = new Word2Vec()
      word2vec.setNumPartitions(numPartition)
      word2vec.setMinCount(minCount)
      word2vec.setVectorSize(vectorSize)

      val model = word2vec.fit(input)

      model.save(sc, modelPath)
    } finally {
      sc.stop()
    }

    println("Elapsed time: " + "%,d".format(System.nanoTime - startTime) + " ns")
  }
}
