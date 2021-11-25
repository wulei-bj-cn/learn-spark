package geek.spark.kafka


import java.lang.management.ManagementFactory
import java.lang.reflect.Modifier
import java.util.Properties


import org.apache.kafka.clients.producer.{Callback, KafkaProducer, ProducerConfig, ProducerRecord, RecordMetadata}
import org.apache.kafka.common.serialization.StringSerializer


object LocalUsageMonitor {
  def getUsage(mothedName: String): Any = {
    val operatingSystemMXBean = ManagementFactory.getOperatingSystemMXBean
    for (method <- operatingSystemMXBean.getClass.getDeclaredMethods) {
      method.setAccessible(true)
      if (method.getName.startsWith(mothedName) && Modifier.isPublic(method.getModifiers)) {
        return method.invoke(operatingSystemMXBean)
      }
    }
    throw new Exception(s"cannot get the usage of ${mothedName}")
  }


  def getMemoryUsage(): String = {
    var freeMemory = 0L
    var totalMemory = 0L
    var usage = 0.0
    try{
      freeMemory = getUsage("getFreePhysicalMemorySize").asInstanceOf[Long]
      totalMemory = getUsage("getTotalPhysicalMemorySize").asInstanceOf[Long]
      usage = (totalMemory - freeMemory.doubleValue) / totalMemory * 100
    } catch {
      case e: Exception => throw e
    }
    usage.toString
  }


  private def getCPUUsage(): String = {
    var usage = 0.0
    try{
      usage = getUsage("getSystemCpuLoad").asInstanceOf[Double] * 100
    } catch {
      case e: Exception => throw e
    }
    usage.toString
  }


  def initConfig(clientID: String): Properties = {
    val props = new Properties
    val brokerList = "localhost:9092"
    props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, brokerList)
    props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, classOf[StringSerializer].getName)
    props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, classOf[StringSerializer].getName)
    props.put(ProducerConfig.CLIENT_ID_CONFIG, clientID)
    props
  }


  val callback = (metadata: RecordMetadata, exception: Exception) => {
    def onCompletion(metadata: RecordMetadata, exception: Exception) = if (exception != null) exception.printStackTrace()
    else println(metadata.topic + "-" + metadata.partition + ":" + metadata.offset)


    onCompletion(metadata, exception)
  }


  class UsageCallback extends Callback {
    def onCompletion(metadata: RecordMetadata, exception: Exception): Unit = {
      if (exception != null) exception.printStackTrace()
      else println(metadata.topic + "-" + metadata.partition + ":" + metadata.offset)
    }
  }


  def main(args: Array[String]): Unit = {
    var clientID = "usage.monitor.client"
    val cpuTopic = "cpu-monitor"
    val memTopic = "mem-monitor"
    if (args.length != 1) {
      System.err.println("Argument 0 must be client id.")
      System.exit(1)
    } else clientID = args(0)


    val props = initConfig(clientID)
    val producer = new KafkaProducer[String, String](props)
    val usageCallback = new UsageCallback()


    while (true) {
      var cpuUsage = new String
      var memoryUsage = new String
      try{
        cpuUsage = getCPUUsage()
        memoryUsage = getMemoryUsage()
        println(s"cpuUsage: ${cpuUsage}, memoryUsage: ${memoryUsage}")
      } catch {
        case e: Exception => {
          System.err.println(e.toString)
          System.exit(1)
        }
      }
      val cpuRecord = new ProducerRecord[String, String](cpuTopic, clientID, cpuUsage)
      val memRecord = new ProducerRecord[String, String](memTopic, clientID, memoryUsage)
      producer.send(cpuRecord, usageCallback)
      producer.send(memRecord, usageCallback)
      Thread.sleep(2000)
    }
  }
}

