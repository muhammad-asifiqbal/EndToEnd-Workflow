import pyspark as spark
from pyspark.sql import SparkSession
import seaborn as sns
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName('Practise').getOrCreate()

class Datasets:
   def dataImport(self):
        # Import Datasets
        df_pyspark1 = spark.read.csv('winequality-red.csv',header=True,inferSchema=True)
        df_pyspark2 = spark.read.csv('winequality-white.csv',header=True,inferSchema=True)
        # Adding additional Column
        df1=df_pyspark1.withColumn('is_red',df_pyspark1['quality']/df_pyspark1['quality'])
        df2=df_pyspark2.withColumn('is_red',df_pyspark2['quality']*0)
        # Merging Two Datasets
        data = df1.union(df2)
        # Renaming Columns
        data=data.withColumnRenamed('fixed acidity','fixed_acidity')\
            .withColumnRenamed('volatile acidity','volatile_acidity')\
            .withColumnRenamed('citric acid','citric_acid')\
            .withColumnRenamed('residual sugar','residual_sugar')\
            .withColumnRenamed('free sulfur dioxide','free_sulfur_dioxide')\
            .withColumnRenamed('total sulfur dioxide','total_sulfur_dioxide')\
            .withColumnRenamed('pH','_pH')
        # Spark DataFrame into Pandas DataFrame
        PandaDF=data.toPandas()

        return PandaDF
        
#Df= Datasets()
#data=Df.dataImport()
#data.describe('column').show()
#PandaDF=data.toPandas()
#print(data)
