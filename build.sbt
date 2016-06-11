name := "word2vec-model-generator"
version := "0.0.1"
scalaVersion := "2.11.8"
libraryDependencies ++=
  Seq("org.apache.spark" % "spark-core_2.11" % "1.6.1" % "provided",
      "org.apache.spark" % "spark-mllib_2.11" % "1.6.1" % "provided")
assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
