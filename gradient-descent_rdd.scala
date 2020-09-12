// Databricks notebook source
import scala.collection.mutable.ArrayBuffer
import scala.math.sqrt
import breeze.linalg.{norm, DenseVector => BDV}


// COMMAND ----------

val dataset=(1 to 1000).toArray
val brace=dataset.map(x=>(ArrayBuffer(x.asInstanceOf[Double],(x+1).asInstanceOf[Double]),(5*x+2).asInstanceOf[Double]))



// COMMAND ----------

def full(v:ArrayBuffer[Double], w:ArrayBuffer[Double])=
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
full(v,w)


// COMMAND ----------

def block_by_scal(v:ArrayBuffer[Double],w:Double)={
  
  var z=ArrayBuffer[Double]()
  val n=v.size
  
  for (i<-0 until n )
     { 
       z+=v(i)*w
     }
   z
}

block_by_scal(v,10)


// COMMAND ----------

def block_scal (x:ArrayBuffer[Double], y:ArrayBuffer[Double]):Double =
{
  var full=0.0
  val T=(x,y).zipped map(_*_) 
  
  for (i<-0 until T.length)
  
  {
    full+=T(i)
  }
  full
}

block_scal(v,w)

// COMMAND ----------

//substract two arrays pulled out of a matrix (Array[Array[Int]])
def defect(v:ArrayBuffer[Double],w:ArrayBuffer[Double])=
{ 
  var z=ArrayBuffer[Double]()
  for (i<-0 until w.length )
     { 
       z+=v(i)-w(i)
     }
   z
}


// COMMAND ----------

//calcul du gradient:this method must return a tuple of the function  
def Gradient(v:(ArrayBuffer[Double],Double),u:ArrayBuffer[Double])=
{
    
  var f=2*(block_scal(v._1,u)-v._2)
  var grad=block_by_scal(v._1,f) 
    
   grad 
  
}
var z = ArrayBuffer(3.0,1.0,5.0)
val v = ArrayBuffer(1.0,2.0,3.0)
Gradient((v,3),z)  


// COMMAND ----------

//GD Runs gradient descent on the given training data/ Gradient object (used to compute the gradient of the loss function)
def GD( p:Array[(ArrayBuffer[Double] , Double)],W:ArrayBuffer[Double] )={
  var visionary_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}

  for (i <- p)
  {
    var grad = Gradient(i,visionary_W)
    gradient = full(gradient , grad )
    
  }
  
  gradient
}
//Class used to solve an optimization problem using Gradient Descent.
val rdds=sc.parallelize(pairs,1).repartition(num_part)
val panel=rdds.glom.zipWithIndex
val count_part=10
var viz=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch = 20

 
var gradient = new ArrayBuffer[Double](nx.length)
for (j<-1 to nx.length ){gradient+=0.0}
var visionary_W=viz

for (i<-1 to epoch)
{
  for(i<- 0 to count_part-1)
  {

    val currpart=panel.filter(p=>p._2==i)
    val grad=currpart.map(x=>GD( x._1 ,viz )).collect
    gradient = full(gradient , grad(0) )
  }
  
  gradient=block_by_scal(gradient,1/(pairs.length.toFloat))
  visionary_W =  defect(visionary_W ,block_by_scal(gradient,eta )) 
  viz=visionary_W
  println(viz)
}

// COMMAND ----------

//SGD require a higher computation effort to process each observation
def SGD( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double , W:ArrayBuffer[Double] )={
  var visionary_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (i <- p)
  {
    var grad = Gradient(i,visionary_W)
    visionary_W =  defect(visionary_W ,block_by_scal(grad,eta )) 
  }
  
  visionary_W
}

val count_part=10
val rdds=sc.parallelize(pairs,1).repartition(num_part)
val panel=rdds.glom.zipWithIndex
var viz=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch = 3

for (i<-1 to epoch)
{
  for(i<- 0 to num_part-1)
  {

    val currpart=partitions.filter(p=>p._2==i)
    val nvar=currpart.map(x=>SGD( x._1 , eta , viz )).collect
    viz=nvar(0)
    println(viz)
  }
}

// COMMAND ----------

//SGD MINI BATCH we sample a subset  of the total data in order to compute a gradient estimate
val rdds=sc.parallelize(pairs,1).repartition(10)
val panel=rdds.glom.zipWithIndex
val mini_batch_rate=10


var nw=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val epoch=3

def SGD_miniBatch( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double , W:ArrayBuffer[Double] )={
  var visionary_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}

  for (i <- p)
  {
    var grad = Gradient(i,visionary_W)
    gradient = full(gradient , grad )
    
  }
  
  gradient=block_by_scal(gradient,1/(p.length.toFloat))
  visionary_W =  defect(visionary_W ,block_by_scal(gradient,eta ))
  visionary_W
}


for (j<-1 to epoch)
 {
   
  for(i<- 0 to mini_batch_rate-1){

    val currpart=partitions.filter(p=>p._2==i)
    val nvar=currpart.map(x=>SGD_miniBatch( x._1 , eta , nx )).collect
    viz=nvar(0)
    println(viz)
  }
}

// COMMAND ----------

//MOMENTUM
def Momentum( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double, Teta:Double, V:ArrayBuffer[Double], W:ArrayBuffer[Double] )=
{
  var visionary_v= V
  var visionary_W= W
  
  var gradient = new ArrayBuffer[Double](W.length)
  for (j<-1 to W.length){gradient+=0.0}
  
  
  for (i <- p)
  {
    var grad = Gradient(i,visionary_W)
    visionary_v=full(block_by_scal(visionary_V,Teta),block_by_scal(grad,eta))
    visionary_W=full(visionary_W ,visionary_V)
    
  }
  
  (visionary_W,visionary_V)
}

val rdds=sc.parallelize(pairs,1).repartition(10)
val panel=rdds.glom.zipWithIndex
val num_part=10

var viz=ArrayBuffer(0.0,0.0)
val eta = 0.00000025
val Teta=0.9

var visionary_V = ArrayBuffer(0.0,0.0)

val epoch=3
for (j<-1 to epoch)
{
  
    for(i<- 0 to num_part-1)
  {

      val currpart=partitions.filter(p=>p._2==i)
      val nvar=currpart.map(x=>Momentum( x._1 , eta ,Teta , visionary_V , viz )).collect
      viz=nvar(0)._1
      visionary_V=nvar(0)._2
      println(nw)
   }
}

// COMMAND ----------

//An implementation of Adagrad

val rdds=sc.parallelize(pairs,1).repartition(10)
val panel=rdds.glom.zipWithIndex
val num_part=10

var viz=ArrayBuffer(0.0,0.0)
val eta = 0.0025
var weightDecay=eta

def Adagrad( p:Array[(ArrayBuffer[Double] , Double)] , eta:Double,  W:ArrayBuffer[Double] )=
{
  var visionary_W  = W
  var gradient = new ArrayBuffer[Double](W.length)
  for (i <- p)
  {
    
    var grad = Gradient(i,visionary_W)
    var learningRate=block_scal(grad,grad)+ 1e-8
    var weightDecay = eta * (1/sqrt(learningRate))
    visionary_W=full(visionary_W,block_by_scal(grad,weightDecay))
  }
  
  visionary_W
}




val epoch=3
for (j<-1 to epoch)
{
  for(i<- 0 to num_part-1)
  {

      val currpart=panel.filter(p=>p._2==i)
      val nvar=currpart.map(x=>Adagrad( x._1 , eta , viz )).collect
      viz=nvar(0)
      println(viz)
  }

}



// COMMAND ----------


