/**
2021-09-20
*/

import org.apache.spark.sql.DataFrame

// val rootPath: String = _
val rootPath: String = "/Users/wulei/Raymond/git/spark-intro2/22 从房价预测开始/data/house-prices-advanced-regression-techniques"
val filePath: String = s"${rootPath}/train.csv"

val trainDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)

// 所有数值型字段
val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea")

// 所有非数值型字段
val categoricalFields: Array[String] = Array("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition")

// Label字段
val labelFields: Array[String] = Array("SalePrice")

// 抽取所有数值型字段
var selectedFields: DataFrame = trainDF.selectExpr((numericFields ++ categoricalFields ++ labelFields): _*)

// =======================处理非数值型特征=======================

import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer

val indexFields: Array[String] = categoricalFields.map(_ + "Index").toArray

for ((field, indexField) <- categoricalFields.zip(indexFields)) {
  val indexer = new StringIndexer()
    .setInputCol(field)
    .setOutputCol(indexField)
  selectedFields = indexer.fit(selectedFields).transform(selectedFields)
  selectedFields = selectedFields.drop(field)
}

val oheFields: Array[String] = categoricalFields.map(_ + "OHE").toArray

for ((field, oheField) <- indexFields.zip(oheFields)) {
  val oheEncoder = new OneHotEncoder()
    .setInputCol(field)
    .setOutputCol(oheField)
  selectedFields = oheEncoder.transform(selectedFields)
  selectedFields = selectedFields.drop(field)
}

// =======================处理非数值型特征=======================

import org.apache.spark.sql.types.IntegerType

var typedFields = selectedFields

for (field <- (numericFields ++ labelFields)) {
  typedFields = typedFields.withColumn(s"${field}Int",col(field).cast(IntegerType)).drop(field)
}

import org.apache.spark.ml.feature.VectorAssembler

val features: Array[String] = numericFields.map(_ + "Int").toArray
val assembler = new VectorAssembler().setInputCols(features ++ oheFields).setOutputCol("features").setHandleInvalid("skip")

var featuresAdded: DataFrame = assembler.transform(typedFields)
for (feature <- features) {
  featuresAdded = featuresAdded.drop(feature)
}

val Array(trainSet, testSet) = featuresAdded.randomSplit(Array(0.7, 0.3))

import org.apache.spark.ml.regression.LinearRegression

// val lr = new LinearRegression().setLabelCol("SalePriceInt").setFeaturesCol("features").setMaxIter(10)
val lr = new LinearRegression().setLabelCol("SalePriceInt").setFeaturesCol("features").setMaxIter(1000)
val lrModel = lr.fit(trainSet)

val trainingSummary = lrModel.summary
println(s"Root Mean Squared Error (RMSE) on train data: ${trainingSummary.rootMeanSquaredError}")
// RMSE: 38288.77947156114

import org.apache.spark.ml.evaluation.RegressionEvaluator

val predictions: DataFrame = lrModel.transform(testSet).select("SalePriceInt", "prediction")
val evaluator = new RegressionEvaluator().setLabelCol("SalePriceInt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

