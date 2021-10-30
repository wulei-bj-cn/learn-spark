/**
2021-08-28

案例地址：https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data


代码：
*/

import org.apache.spark.sql.DataFrame

// val rootPath: String = _
val rootPath: String = "/Users/wulei/Raymond/git/spark-intro2/22 从房价预测开始/data/house-prices-advanced-regression-techniques"
val filePath: String = s"${rootPath}/train.csv"

val numericFields: Array[String] = Array("LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea")
val labelFields: Array[String] = Array("SalePrice")

val trainDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)

import org.apache.spark.sql.types.IntegerType

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

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression().setLabelCol("SalePriceInt").setFeaturesCol("features").setMaxIter(10)
val lrModel = lr.fit(trainSet)

val trainingSummary = lrModel.summary
println(s"Root Mean Squared Error (RMSE) on train data: ${trainingSummary.rootMeanSquaredError}")

import org.apache.spark.ml.evaluation.RegressionEvaluator

val predictions: DataFrame = lrModel.transform(testSet).select("SalePriceInt", "prediction")
val evaluator = new RegressionEvaluator().setLabelCol("SalePriceInt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

