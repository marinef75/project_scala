package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.log4j.{Level, Logger}


object Trainer {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("BFGS").setLevel(Level.ERROR)

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


    /*******************************************************************************
      *
      *       TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

   /** CHARGER LE DATASET **/

     //chargement du dataframe
    val df = spark.read.parquet("/Users/marine/Documents/Telecom/spark/TP_parisTech_2017_2018_starter/data/prepared_trainingset")

    df.printSchema


    /** TF-IDF **/
      // stage 1 : tokenization
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val df_tok = tokenizer.transform(df)
    df_tok.show(5)

    // stage 2 : on élimine les stopwords
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered_tokens")


    // stage 3 décompte des mots
    val count_vectorizer= new CountVectorizer()
      .setInputCol("filtered_tokens")
      .setOutputCol("tf")

    //stage4  calcul de l'idf
    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")

    val df_tok_rem = remover.transform(df_tok).drop("tokens")
    val df_tf = count_vectorizer.fit(df_tok_rem).transform(df_tok_rem)
    val df_tfidf = idf.fit(df_tf)transform(df_tf)

    df_tfidf.select("tfidf").show()

    /** VECTOR ASSEMBLER **/

      // conversion des variables catégorielles en variables numériques.

      //stage5 : traitement de la variable catégorielle country2
    val indexer = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("country2")
      .setOutputCol("country_indexed")


    val df_preindexed = indexer.fit(df_tfidf).transform(df_tfidf)
    df_preindexed.groupBy("country2").min("country_indexed").show

    //stage6 traitement de la variable catégorielle currency2
    val indexer2 = new StringIndexer()
      .setHandleInvalid("skip")
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    val df_indexed = indexer2.fit(df_preindexed).transform(df_preindexed)
    df_indexed.groupBy("currency2").min("currency_indexed").show

    df_indexed.printSchema()

    // stage7 :  on assemble toutes les colonnes en une colonne contenant des vecteurs.
    val assembler = new VectorAssembler()
      .setInputCols(Array[String]( "tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed" ))
      .setOutputCol("features")


    /** MODEL **/

  // stage 8 création et paramétrage du classifieur par régression logistique.
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      //.setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)



    /** PIPELINE **/

    // création du pipeline

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,remover, count_vectorizer,idf, indexer, indexer2, assembler,lr))


    /** TRAINING AND GRID-SEARCH **/

      //découpage du dataframe en training et test.
    val Array(training, test) = df.randomSplit(Array[Double](0.9, 0.1))

    //paramétrage de la grille de validation croisée pour les paramètres regParam et minDF
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array[Double](10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(count_vectorizer.minDF,  Array[Double](55, 75, 95))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    // création du cross validator

    val cv = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    // entraînement du modèle avec les paramètres optimaux
    val model = cv.fit(training)

    // on enregistre le modèle entraîné dans un fichier binaire pour ppwdouvoir le réutiliser sans avoir à le recalculer.
    model.write.overwrite().save("/Users/marine/Documents/Telecom/spark/TP_parisTech_2017_2018_starter/data/trained_model")

    // prediction du modèle pré-entraîné sur les données de test et visualisation des résultats
    val df_predictions = model.transform(test)
    println("f_mesure = " + evaluator.setMetricName("f1").evaluate(df_predictions))

    // visualisation des faux néfatifs, faux positifs, vrais positifs et vrais négatifs
    df_predictions.groupBy("final_status", "predictions").count.show()

  }
}
