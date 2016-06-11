package jp.muziyoshiz.word2vec

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import java.io.File

object Word2VecModelGenerator {

  def main(args: Array[String]) {
    require (args.length == 2, "argument: <textfilepath> <modelpath>")

    val textFilePath = args(0)
    val modelPath = args(1)
    if (new File(modelPath).exists) {
      println("ERROR: Cannot overwrite existing model \"" + modelPath + "\"")
      System.exit(1)
    }

    val startTime = System.nanoTime

    val conf = new SparkConf().setAppName("Word2VecModelGenerator")
    val sc = new SparkContext(conf)

    try {
      val input = sc.textFile(textFilePath).map(line => line.split(" ").toSeq)
      val word2vec = new Word2Vec()
      val model = word2vec.fit(input)
      model.save(sc, modelPath)
    } finally {
      sc.stop()
    }

    println("Elapsed time: " + "%,d".format(System.nanoTime - startTime) + " ns")
  }
}
