// 2021-09-19

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

import org.apache.spark.ml.feature.VectorAssembler

val features: Array[String] = numericFields.map(_ + "Int").toArray
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features").setHandleInvalid("skip")

var featuresAdded: DataFrame = assembler.transform(typedFields)
for (feature <- features) {
  featuresAdded = featuresAdded.drop(feature)
}

val Array(trainSet, testSet) = featuresAdded.randomSplit(Array(0.7, 0.3))

// ==========================卡方检验==========================

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.feature.ChiSqSelectorModel

val selector = new ChiSqSelector().setNumTopFeatures(20).setFeaturesCol("features").setLabelCol("SalePriceInt").setOutputCol("selectedFeatures")

val chiSquare = selector.fit(trainSet)
val indexs: Array[Int] = chiSquare.selectedFeatures

import scala.collection.mutable.ArrayBuffer

val selectedFeatures: ArrayBuffer[String] = ArrayBuffer[String]()
for (index <- indexs) {
  selectedFeatures += numericFields(index)
}

// ==========================卡方检验==========================

import org.apache.spark.sql.DataFrame

// val rootPath2: String = _
val rootPath2: String = "/Users/wulei/Raymond/git/spark-intro2/22 从房价预测开始/data/house-prices-advanced-regression-techniques"
val filePath2: String = s"${rootPath2}/train.csv"

val trainDF2: DataFrame = spark.read.format("csv").option("header", true).load(filePath2)

// 所有数值型字段
/**
val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea")
*/
val numericFields2: Array[String] = selectedFeatures.toArray
val labelFields2: Array[String] = Array("SalePrice")

// 抽取所有数值型字段
val selectedFields2: DataFrame = trainDF.selectExpr((numericFields2 ++ labelFields2): _*)

import org.apache.spark.sql.types.IntegerType

var typedFields2 = selectedFields2

for (field <- (numericFields2 ++ labelFields2)) {
  typedFields2 = typedFields2.withColumn(s"${field}Int",col(field).cast(IntegerType)).drop(field)
}

import org.apache.spark.ml.feature.VectorAssembler

val features2: Array[String] = numericFields2.map(_ + "Int").toArray
val assembler2 = new VectorAssembler().setInputCols(features2).setOutputCol("features").setHandleInvalid("skip")

var featuresAdded2: DataFrame = assembler2.transform(typedFields2)
for (feature <- features2) {
  featuresAdded2 = featuresAdded2.drop(feature)
}

val Array(trainSet2, testSet2) = featuresAdded2.randomSplit(Array(0.7, 0.3))

import org.apache.spark.ml.regression.LinearRegression

val lr2 = new LinearRegression().setLabelCol("SalePriceInt").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
// val lr = new LinearRegression().setLabelCol("SalePriceInt").setFeaturesCol("features").setMaxIter(1000)
val lrModel2 = lr2.fit(trainSet2)

val trainingSummary2 = lrModel2.summary
println(s"Root Mean Squared Error (RMSE) on train data: ${trainingSummary2.rootMeanSquaredError}")
// RMSE: 38288.77947156114

import org.apache.spark.ml.evaluation.RegressionEvaluator

val predictions2: DataFrame = lrModel2.transform(testSet2).select("SalePriceInt", "prediction")
val evaluator2 = new RegressionEvaluator().setLabelCol("SalePriceInt").setPredictionCol("prediction").setMetricName("rmse")
val rmse2 = evaluator2.evaluate(predictions2)
println("Root Mean Squared Error (RMSE) on test data = " + rmse2)
