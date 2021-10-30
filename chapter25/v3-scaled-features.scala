/**
2021-09-19

*/

import org.apache.spark.sql.DataFrame

// val rootPath: String = _
val rootPath: String = "/Users/wulei/Raymond/git/spark-intro2/22 从房价预测开始/data/house-prices-advanced-regression-techniques"
val filePath: String = s"${rootPath}/train.csv"

val trainDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)

// 所有数值型字段
val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea")
val labelFields: Array[String] = Array("SalePrice")

// 抽取所有数值型字段
val selectedFields: DataFrame = trainDF.selectExpr((numericFields ++ labelFields): _*)

import org.apache.spark.sql.types.IntegerType

var typedFields = selectedFields

for (field <- (numericFields ++ labelFields)) {
  typedFields = typedFields.withColumn(s"${field}Int",col(field).cast(IntegerType)).drop(field)
}

// =============================转为Vector==================================

var vectorsData: DataFrame = typedFields
import org.apache.spark.ml.feature.VectorAssembler

val featureFields: Array[String] = numericFields.map(_ + "Int").toArray

for (field <- featureFields) {
  val assembler = new VectorAssembler().setInputCols(Array(field)).setOutputCol(s"${field}Vector").setHandleInvalid("skip")
  vectorsData = assembler.transform(vectorsData)
  vectorsData = vectorsData.drop(field)
}

// =============================归一化==================================

var scaledData: DataFrame = vectorsData

import org.apache.spark.ml.feature.MinMaxScaler

val vectorFields: Array[String] = featureFields.map(_ + "Vector").toArray

for (vector <- vectorFields) {
  val minMaxScaler = new MinMaxScaler()
   .setInputCol(vector)
   .setOutputCol(s"${vector}Scaled")

  val scalerModel = minMaxScaler.fit(scaledData)
  scaledData = scalerModel.transform(scaledData)
  scaledData = scaledData.drop(vector)
}

// ===========================最后捏合为一个vector==============================
val scaledFields: Array[String] = vectorFields.map(_ + "Scaled").toArray
var trainingSamples = scaledData

val assembler = new VectorAssembler().setInputCols(scaledFields).setOutputCol("features").setHandleInvalid("skip")

trainingSamples = assembler.transform(trainingSamples)
for (field <- scaledFields) {
  trainingSamples = trainingSamples.drop(field)
}

// ===========================训练 & 预测 ==============================

val Array(trainSet, testSet) = trainingSamples.randomSplit(Array(0.7, 0.3))

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression().setLabelCol("SalePriceInt").setFeaturesCol("features").setMaxIter(10)
val lrModel = lr.fit(trainSet)

val trainingSummary = lrModel.summary
println(s"Root Mean Squared Error (RMSE) on train data: ${trainingSummary.rootMeanSquaredError}")
// RMSE: 38288.77947156114

import org.apache.spark.ml.evaluation.RegressionEvaluator

val predictions: DataFrame = lrModel.transform(testSet).select("SalePriceInt", "prediction")
val evaluator = new RegressionEvaluator().setLabelCol("SalePriceInt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

