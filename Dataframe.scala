// Databricks notebook source
// import packages
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.functions.spark_partition_id

// COMMAND ----------

//Define class row
case class row(v1 : Double, v2:Double , w:Double)

//Generate Dataframe
val Dataset=for (i <- 1 to 1000) yield (row(i.asInstanceOf[Double],(i+5).asInstanceOf[Double],(5*i+2).asInstanceOf[Double]))
val Df_frame=Dataset.toDF()
Df_frame .show()

// COMMAND ----------

//partition number
val DataF = Df_frame.repartition(10)
var DataF_f =DF.withColumn("partition_ID", spark_partition_id)
DataF_f.show()

// COMMAND ----------

val v = ArrayBuffer(10.0,55.0,5.0)
val w= ArrayBuffer(5.0,5.0,1.0)

def block_by_scal(v:ArrayBuffer[Double],n:Double)={
  val s=v.size
  var med=ArrayBuffer[Double]()
  for(i<- 0 until s){
    med+=v(i)*n
  }
  med
}

def block_scal (v:ArrayBuffer[Double], u:ArrayBuffer[Double]):Double =
{
  (v zip u).map(vw=>vw._1*vw._2).sum  
}

block_by_scal(v,10)
block_scal(v,w)

// COMMAND ----------

def diff(v:ArrayBuffer[Double],w:ArrayBuffer[Double])={ 
  val s=v.size
  var med=ArrayBuffer[Double]()
  for (i<-0 until s)
     { 
       med+=v(i)-w(i)
     }
   med
}
def sommation(v:ArrayBuffer[Double],w:ArrayBuffer[Double])={
  val s=v.size
  var med=new ArrayBuffer[Double]()
  for(i<-0 until s){
    med+=v(i)+w(i)
  }
  med
}

sommation(v,w)
diff(v,w)

// COMMAND ----------

//calculating the gradient of a single instance
def grad(v:(ArrayBuffer[Double],Double),u:ArrayBuffer[Double])={
   var med=block_by_scal(v._1,2*(block_scal(v._1,u)-v._2)) 
   med  
}

// COMMAND ----------

//Gradient  with Batch
var pip=ArrayBuffer.fill(2)(0.0)
var word_wine =ArrayBuffer.fill(pip.length)(0.0)
var new_pip=pip
val beta = 0.000001
val iter_num = 10
val parts=10
for (k<-1 to iter_num)
{
  for(i<- 0 to parts-1)
  {
    val part_act=DataF_f.filter($"partition_ID" === i )
    val grad=grad_Batch(part_act ,pip )
    word_wine = sommation(word_wine , grad )
  }
  
  word_wine=block_by_scal(word_wine,1/(DataF_f.count.toFloat))
  new_pip =  diff(new_pip ,block_by_scal(word_wine,beta )) 
  pip=new_pip
  println(pip)
}
//gradient partition(Batch)
def grad_Batch( dataframe:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row],u:ArrayBuffer[Double] )={
  var grad_par = ArrayBuffer.fill(u.length)(0.0)
  var row=Df_frame.select($"v1",$"v2",$"w").collect()
  for (j<-row){ 
    var v1=j.get(0).asInstanceOf[Double]
    var v2=j.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =j.get(2).asInstanceOf[Double]
    grad_par= sommation(grad_par ,  grad((v,w),u) ) 
  }
  grad_par
}



// COMMAND ----------

//(SGD)
val parts=10
var pip=ArrayBuffer.fill(2)(0.0)
val beta = 0.000001
val num_iter = 3
for (k<-1 to num_iter){
  for(i<- 0 to parts-1){
    val part_act=DataF_f.filter($"partition_ID" === i )
    val new_pip=SGD( part_act , beta , pip )
    pip=new_pip
    println(new_pip)
 }   

}
//SGD-Parition
def SGD( Df_frame:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , beta:Double , u:ArrayBuffer[Double] )={
  var new_pip  = pip
  var row=Df_frame.select($"v1",$"v2",$"w").collect()
  for (j<-row){ 
    var v1=j.get(0).asInstanceOf[Double]
    var v2=j.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =j.get(2).asInstanceOf[Double]    
    var grad_inst = grad((v,w),new_pip)
    new_pip =  diff(new_pip ,block_by_scal(grad_inst,beta )) 
  }
  new_pip
}



// COMMAND ----------

// SGD  with MiniBatch
var pip=ArrayBuffer.fill(2)(0.0)
val beta = 0.000001
val num_iter=3
val minibatch_pace=30

for (j<-1 to num_iter){
  for(i<- 0 to parts-1){
    val part_act=DataF_f.filter($"partition_ID" === i )
    var samples=part_act.sample(true,minibatch_pace)
    val new_pip=SGD_miniBatch( samples , beta , pip )
    pip=new_pip
    println(pip)
  }
}
// MiniBatch-Partition
def SGD_miniBatch( Df_frame:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , beta:Double , u:ArrayBuffer[Double] )={
  var new_pip  = pip
  var grad_Batch = ArrayBuffer.fill(pip.length)(0.0) 
  var rows=Df_frame.select($"v1",$"v2",$"w").collect()
  for (j<-rows){ 
    var v1=j.get(0).asInstanceOf[Double]
    var v2=j.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =j.get(2).asInstanceOf[Double]    
    var grad_inst = grad((v,w),new_pip)
    grad_Batch= sommation(grad_Batch , grad_inst )
  }
  grad_Batch=block_by_scal(grad_Batch,1/(Df_frame.count.toFloat))
  new_pip =  diff(new_pip ,block_by_scal(grad_Batch,beta ))
  new_pip
}



// COMMAND ----------

//MOMENTUM-Partition
def SGD_Momentum( Df_frame:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , beta:Double, gamma:Double, pip:ArrayBuffer[Double], pip1:ArrayBuffer[Double] )={
  var new_pip  = pip
  var new_pip1 = pip1
  var gradient = ArrayBuffer.fill(pip.length)(0.0) 
  var rows=Df_frame.select($"v1",$"v2",$"w").collect()
  for (i<-rows){ 
    var v1=i.get(0).asInstanceOf[Double]
    var v2=i.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =i.get(2).asInstanceOf[Double]    
    var grad_inst = grad((v,w),new_pip)
    new_pip1=sommation(block_by_scal(new_pip1,gamma),block_by_scal(grad_inst,beta))
    new_pip =  diff(new_pip ,new_pip1)
  }
  (new_pip,new_pip1)
}

//Fast SGD with Momentum
var new_pip=ArrayBuffer.fill(2)(0.0)
val beta = 0.000001
val gamma=0.9

var new_pip1 =ArrayBuffer.fill(2)(0.0)
val num_iter=3
for (k<-1 to num_iter){
    for(i<- 0 to parts-1 ){
      val part_act=DF_f.filter($"partition_ID" === i )
      val (nvvpip,nvvpip1)=SGD_Momentum( part_act , beta ,gamma , new_pip , new_pip1)
      new_pip=nvvpip
      new_pip1=nvvpip1
      println(new_pip)
   }
}


// COMMAND ----------

// Fast SGD with Adagrad 
var new_pip=ArrayBuffer.fill(2)(0.0)
val beta = 0.0025
var new_beta=beta

val num_iter=3
for (j<-1 to num_iter)
{
  for(i<- 0 to parts-1){
      val part_act=DataF_f  .filter($"partition_ID" === i )
      val nvvpip=SGD_Adagrad( part_act , beta , new_pip )
      new_pip=nvvpip
      println(new_pip)
  }
}
//ADAGRAD-Partition
import scala.math.sqrt
def SGD_Adagrad( Df_frame:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , beta:Double,  pip:ArrayBuffer[Double] )=
{
  var new_pip  = pip
  var gradient = new ArrayBuffer[Double](pip.length)
  
  var rows=Df_frame.select($"v1",$"v2",$"w").collect()
  for (i<-rows){ 
    var v1=i.get(0).asInstanceOf[Double]
    var v2=i.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =i.get(2).asInstanceOf[Double]    
    var grad_inst = grad((v,w),new_pip)
    var ADA=block_scal(grad_inst,grad_inst)+ 1e-8
    var new_beta = beta * (1/sqrt(ADA))
    new_pip=diff(new_pip,block_by_scal(grad_inst ,new_beta))
  }
  new_pip
}



// COMMAND ----------


