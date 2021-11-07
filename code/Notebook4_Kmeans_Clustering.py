# Databricks notebook source
# MAGIC %md
# MAGIC #Taste ML

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to load objects defined in Notebook1

# COMMAND ----------

#TODO: prep the data
fac_coords_df = spark.read.table("data_modified_marco_for_marco_2_csv")
display(fac_coords_df)

# COMMAND ----------

# MAGIC %md
# MAGIC -------------------------------- **Above will be outputs from runing Notebook1** --------------------------------------

# COMMAND ----------

# Check that all desired objects from Notebook1 are loaded in this Notebook
dir()

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark ML KMeans Clustering

# COMMAND ----------

# extract facility coordinates from facility_df DataFrame
fac_coords_df = (fac_coords_df
                 .select('Username', 'Recipe'))


# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator
fac_coords_df = fac_coords_df.groupBy("Username", "Recipe").count()
display(fac_coords_df)



# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(inputCols=["count"], outputCol="features")
fac_coords = vecAssembler.transform(fac_coords_df)


# COMMAND ----------

fac_coords.show(5, truncate=False)

# COMMAND ----------

fac_coords.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 200 Clusters

# COMMAND ----------

kmeans_200 = KMeans().setK(2)
KM200_model = kmeans_200.fit(fac_coords)
fac200_centers = KM200_model.clusterCenters()
print (len(fac200_centers))
print (fac200_centers)

# COMMAND ----------

fac200_centers = np.asarray(fac200_centers)
fac200_centers

# COMMAND ----------

from pyspark.sql.functions import col
fac200_label_df = (KM200_model
                   .transform(fac_coords)
                   .select(col("features").alias("coords"), col("prediction").alias("label")))
fac200_label_df.show(5, truncate=False)
KM200_label= (fac200_label_df
              .select("label")
              .rdd
              .map(lambda x: x['label'])
              .collect())

# COMMAND ----------

print(KM200_label)
print(set(KM200_label))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualization 200 Clusters and their centroids

# COMMAND ----------

fac_lat, fac_lon = zip(*fac_coords_df.selectExpr("Username", "Recipe").collect())
fac_lat

# COMMAND ----------

fac200_centers.shape

# COMMAND ----------

# Plot the faciity clusters and cluster centroid
fig, ax = plt.subplots(figsize=[20, 12])
facility_scatter = ax.scatter(fac_lon, fac_lat, c=KM200_label, cmap = cm.Dark2, edgecolor='None', alpha=0.7, s=120)
centroid_scatter = ax.scatter(fac200_centers[1], fac200_centers[0], marker='x', linewidths=2, c='k', s=30)
ax.set_title('10 Facility Clusters & Facility Centroid', fontsize = 20)
ax.set_xlabel('User', fontsize=24)
ax.set_ylabel('Recipe', fontsize = 24)
ax.legend([facility_scatter, centroid_scatter], ['Facilities', 'Facility Cluster Centroid'], loc='upper right', fontsize = 20)
display(fig)

# COMMAND ----------


