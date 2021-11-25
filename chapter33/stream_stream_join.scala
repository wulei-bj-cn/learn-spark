import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType

val rootPath: String = "./data"

val postSchema = new StructType().add("id", "integer").add("name", "string").add("postTime", "timestamp")
val postStream: DataFrame = spark.readStream.format("csv").option("header", true).option("path", s"${rootPath}/videoPosting").schema(postSchema).load
val postStreamWithWatermark = postStream.withWatermark("postTime", "5 minutes")

val actionSchema = new StructType().add("userId", "integer").add("videoId", "integer").add("event", "string").add("eventTime", "timestamp")
val actionStream: DataFrame = spark.readStream.format("csv").option("header", true).option("path", s"${rootPath}/interactions").schema(actionSchema).load
val actionStreamWithWatermark = actionStream.withWatermark("eventTime", "5 minutes")

val jointDF: DataFrame = actionStreamWithWatermark.join(postStreamWithWatermark, expr("""
    videoId = id AND
    eventTime >= postTime AND
    eventTime <= postTime + interval 1 hour
    """))

jointDF.writeStream

// 指定Sink为终端（Console）
.format("console")

// 指定输出选项
.option("truncate", false)

// 指定输出模式
.outputMode("append")

// 启动流处理应用
.start()
// 等待中断指令
.awaitTermination()
