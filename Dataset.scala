// Databricks notebook source
import scala.collection.mutable.ArrayBuffer


// COMMAND ----------

//our Data
val dataset_base=1000
case class data(v1 : Double, v2:Double , w:Double)

val Did=for (i <- 1 to dataset) yield (data(i.asInstanceOf[Double],(i+1).asInstanceOf[Double],(5*i+2).asInstanceOf[Double]))
val Ssh=Did.toDS()
Ssh.show()


// COMMAND ----------

Ssh.rdd.getNumPartitions

// COMMAND ----------

val viz = Ssh.repartition(12)

// COMMAND ----------

viz.rdd.getNumPartitions

// COMMAND ----------

viz.show()

// COMMAND ----------

import org.apache.spark.sql.functions.spark_partition_id
var Dataset =viz.withColumn("partitionID", spark_partition_id)


// COMMAND ----------

Dataset.show()

// COMMAND ----------


def sommation(v:ArrayBuffer[Double], w:ArrayBuffer[Double])=
{
  var z = ArrayBuffer[Double]()
  for (i <- 0 until v.size)
  {
    z+=v(i)+w(i)
  }
  z

}
val v = ArrayBuffer(1.0,2.0,3.0)
val w= ArrayBuffer(2.0,4.0,1.0)
sommation(v,w)

// COMMAND ----------

def block_by_scal(v:ArrayBuffer[Double],w:Double)={
  val n=a.size
  var z=ArrayBuffer[Double]()
  for (i<-0 until n )
     { 
       z+=v(i)*w
     }
   z
}

block_by_scal(a,10)

def block_scal (v:ArrayBuffer[Double], w:ArrayBuffer[Double]):Double =
{
  val Y=(v,w).zipped map(_*_) 
  var sommation=0.0
  for (i<-0 until Y.length)
  {
    sommation+=Y(i)
  }
  sommation
}

block_scal(v,w)

// COMMAND ----------

//substract two arrays 
def defect(v:ArrayBuffer[Double],w:ArrayBuffer[Double])= 
{ 
  var z=ArrayBuffer[Double]()
  for (i<-0 until b.length )
     { 
       z+=v(i)-w(i)
     }
   z
}

// COMMAND ----------

def Gradient(v:(ArrayBuffer[Double],Double),u:ArrayBuffer[Double])=
{
   var f=2*(block_scal(v._1,u)-v._2)
   var grad=block_by_scal(v._1,f) 
   grad 
  
}

// COMMAND ----------

//SGD
def SGD( Ssh:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double , pip:ArrayBuffer[Double] )={
  var new_pip  = pip
  var TT=Ssh.select($"v1",$"v2",$"w").collect()
  for (i<-TT)

  { 
    var v1=i.get(0).asInstanceOf[Double]
    var v2=i.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((v,w),new_pip)
    new_pip =  defect(new_pip ,block_by_scal(grad,eta )) 
  }
  
  new_pip
}


var zip=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch = 3
val num_partition=10

for (i<-1 to epoch)
{
  for(i<- 0 to num_partition-1)
  {

    val curr_Ssh=Dataset.filter($"partitionID" === i )
    val nzip=SGD( curr_Ssh , eta , zip )
    zip=nzip
    println(zip)
  }
}

// COMMAND ----------

//Mini Batch SGD
def SGD_miniBatch( Ssh:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double , pip:ArrayBuffer[Double] )={
  var new_pip  = pip
  var gradient = new ArrayBuffer[Double](pip.length)
  for (j<-1 to pip.length){gradient+=0.0}
  
  var TT=Ssh.select($"v1",$"v2",$"w").collect()


  for (i<-TT)

  { 
    var v1=i.get(0).asInstanceOf[Double]
    var v2=i.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((v,w),new_pip)
    
    gradient = sommation(gradient , grad )
    
  }
  
  gradient=block_by_scal(gradient,1/(Ssh.count.toFloat))
  new_pip = defect(new_pip ,block_by_scal(gradient,eta ))
  new_pip
}


var zip=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch=3
val mini_batch_rate=10

for (j<-1 to epoch)
 {
   
  for(i<- 0 to mini_batch_rate-1){

    val curr_Ssh=Dataset.filter($"partitionID" === i )
    val nzip=SGD_miniBatch( curr_Ssh , eta , zip )
    zip=nzip
    println(zip)
  }
}

// COMMAND ----------

//Momentum
def Momentum( Ssh:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double, beta:Double, pip1:ArrayBuffer[Double], pip:ArrayBuffer[Double] )=
{
  var new_pip  = pip
  var new_pip1=pip1
  
  var gradient = new ArrayBuffer[Double](pip.length)
  for (j<-1 to pip.length){gradient+=0.0}
  
  
  var TT=Ssh.select($"v1",$"v2",$"w").collect()
  for (i<-TT)

  { 
    var v1=i.get(0).asInstanceOf[Double]
    var v2=i.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((v,w),new_pip)
    new_pip1=sum(block_by_scal(new_pip1,beta),block_by_scal(grad,eta))
    new_pip =  defect(new_pip ,new_pip1)
    
  }
  (new_pip,new_pip1)
}



var zip=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val beta=0.9

var new_pip1 = ArrayBuffer(0.0,0.0)

val epoch=3
for (j<-1 to epoch)
{
  
    for(i<- 0 to 9)
  {

      val curr_Ssh=Dataset.filter($"partitionID" === i )
      val (nzip,nzip1)=Momentum( curr_Ssh , eta ,beta , new_pip1 , zip )
      zip=nzip
      new_pip1=nzip1
      println(nzip)
   }
}

// COMMAND ----------

//Adagrad

import scala.math.sqrt
def Adagrad( Ssh:org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] , eta:Double,  pip:ArrayBuffer[Double] )=
{
  var new_pip  = pip
  var gradient = new ArrayBuffer[Double](pip.length)
  
  var TT=Ssh.select($"v1",$"v2",$"w").collect()
  for (i<-TT)

  { 
    var v1=i.get(0).asInstanceOf[Double]
    var v2=i.get(1).asInstanceOf[Double]
    var v=ArrayBuffer(v1,v2)
    var w =i.get(2).asInstanceOf[Double]    
    var grad = Gradient((v,w),new_pip)
    
    var GOG=block_scal(grad,grad)+ 1e-8
    var new_eta = eta * (1/sqrt(GOG))
    new_pip=defect(new_pip,block_by_scal(grad,new_eta))
  }
  
  new_pip
}


var zip=ArrayBuffer(0.0,0.0)
val eta = 0.0025
var new_eta=eta

val epoch=3
for (j<-1 to epoch)
{
  for(i<- 0 to 9)
  {

      val curr_Ssh=Dataset.filter($"partitionID" === i )
      val nzip=Adagrad( curr_Ssh , eta , zip )
      zip=nzip
      println(zip)
  }

}

// COMMAND ----------


