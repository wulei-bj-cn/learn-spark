import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions.broadcast

val rootPath: String = "./data"

val staticDF: DataFrame = spark.read.format("csv").option("header", true).load(s"${rootPath}/userProfile/userProfile.csv")

val actionSchema = new StructType().add("userId", "integer").add("videoId", "integer").add("event", "string").add("eventTime", "timestamp")
var streamingDF: DataFrame = spark.readStream.format("csv").option("header", true).option("path", s"${rootPath}/interactions").schema(actionSchema).load

streamingDF = streamingDF.withWatermark("eventTime", "30 minutes").groupBy(window(col("eventTime"), "1 hours"), col("userId"), col("event")).count

val jointDF: DataFrame = streamingDF.join(staticDF, streamingDF("userId") === staticDF("id"))

/**
利用广播变量来优化
val bc_staticDF = broadcast(staticDF)
val jointDF: DataFrame = streamingDF.join(bc_staticDF, streamingDF("userId") === bc_staticDF("id"))
*/

jointDF.writeStream

// 指定Sink为终端（Console）
.format("console")

// 指定输出选项
.option("truncate", false)

// 指定输出模式
.outputMode("update")

// 启动流处理应用
.start()
// 等待中断指令
.awaitTermination()

