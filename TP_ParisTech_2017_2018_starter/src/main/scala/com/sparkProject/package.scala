package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}

object TrainerJerome {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** CHARGER LE DATASET **/

    val df: DataFrame = spark
      .read
      .option("header", true) // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column (ne sert pas à grand chose ici, car il met en string et retraiter au e))
      .option("nullValue", "false") // replace strings "false" (that indicates missing data) by null values
      .parquet("/Users/marine/Documents/Telecom/spark/TP_parisTech_2017_2018/data/prepared_trainingset")

    //b) nombre de lignes et colonnes
    //df.show()
    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    //df.printSchema()

    /** TF-IDF **/
    //Stage 1a. La première étape est séparer les textes en mots (ou tokens) avec un tokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    //Stage 2b. On veut retirer les stop words pour ne pas encombrer le modèle avec des mots
    //qui ne véhiculent pas de sens. Créer le 2ème stage avec la classe StopWordsRemover
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("text_filtered")


    //Stage 3c. La partie TF de TF-IDF est faite avec la classe CountVectorizer
    val vectorizer = new CountVectorizer()
      .setInputCol("text_filtered")
      .setOutputCol("vectorized")


    //Stage 4d. On veut écrire l’output de cette étape dans une colonne “tfidf”.
    val idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")



    //Stage 5e. Convertir la variable catégorielle “country2” en données numérique
    val  index_country = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")


    //Stage 6f. Convertir la variable catégorielle “currency2” en données numérique
    val index_currency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")



    /** VECTOR ASSEMBLER **/

    //Stage 7g. Assembler les features "tfidf", "days_campaign",
    // "hours_prepa", "goal", "country_indexed", "currency_indexed"  dans une seule colonne “features”
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign","hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")



    /** MODEL **/
    //Stage 8h. Le modèle de classification, il s’agit d’une régression logistique
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)



    /** PIPELINE **/
    //créer le pipeline en assemblant les 8 stages définis précédemment, dans le bon ordre
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,remover,vectorizer,idf,index_country,index_currency,assembler,lr))




    /** TRAINING AND GRID-SEARCH **/
    //Créer un dataFrame nommé “training” et un autre nommé “test”  à partir du dataFrame
    //chargé initialement de façon à le séparer en training et test sets dans les proportions
    //90%, 10% respectivement

    //val training, test = df.randomSplit(Array[Double](0.9, 0.1))

    val Array(training, test) = df.randomSplit(Array[Double](0.9, 0.1))


    val param_grid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(vectorizer.minDF,  Array[Double](55, 75, 95))
      .build()


    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")


    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(param_grid)
      .setTrainRatio(0.7)


    print("mémoire OK jusqu'ici")

    val model = trainValidationSplit.fit(training)

    val df_with_predictions = model.transform(test)

    println("f1_score = " + evaluator.setMetricName("f1").evaluate(df_with_predictions))

    df_with_predictions.groupBy("final_status", "predictions").count.show()

    //model.write.overwrite().save("TP_SPARK_4&5_model")


    //println(model.bestModel.)
    //println(model.extractParamMap())




















  }


}
