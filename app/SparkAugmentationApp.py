from app import ImageUtils
import pyspark
from pyspark.sql.types import *
import os

os.environ['SPARK_HOME'] ='/Users/srinivas/Desktop/spark-2.4.6-bin-hadoop2.7'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] ='YES'

if __name__ == '__main__':
    ImageFields = ["origin", "height", "width", "nChannels", "mode", "data"]

    ImageSchema = StructType([
        StructField(ImageFields[0], StringType(), True),
        StructField(ImageFields[1], IntegerType(), True),
        StructField(ImageFields[2], IntegerType(), True),
        StructField(ImageFields[3], IntegerType(), True),
        StructField(ImageFields[4], IntegerType(), True),  # OpenCV type: CV_8U in most cases
        StructField(ImageFields[5], BinaryType(), True)])

    image_schema = StructType().add("image",ImageSchema)

    spark = pyspark.sql.SparkSession.builder.appName("SparkAugmentationApp") \
        .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()

    from pyspark.sql.functions import regexp_replace,col

    from mmlspark.opencv import ImageTransformer

    images_input_dir  = "/Users/srinivas/Desktop/github_projects/spark-augmentation-app/images"
    images_output_dir = "/Users/srinivas/Desktop/github_projects/spark-augmentation-app/images_processed"

    images = spark.read.format("image").load(images_input_dir)


    def image_transform_mml():

        img_transformer = ImageTransformer().setOutputCol("img_out").\
            flip(flipCode = 1).\
            resize(height = 200, width = 200).\
            crop(0, 0, height = 180, width = 180)
        transformed_images = img_transformer.transform(images).select("img_out").withColumnRenamed("img_out", "image")
        df_new = transformed_images.withColumn('filenames',
                                               regexp_replace(col("image.origin"), images_input_dir, images_output_dir))
        df_new.write.format("org.apache.spark.ml.source.image.PatchedImageFileFormat").save(images_output_dir)


    def image_transform_augmented(no_of_augmented_images):

        img_transformer = ImageTransformer().setOutputCol("out")

        transformed_images = img_transformer.transform(images).select("out").withColumnRenamed("out", "image")
        transformed_df = transformed_images.select("image.height", "image.width", "image.nChannels", "image.mode", "image.origin", "image.data",
                                  "image.origin")

        augement_imgs_df_new =transformed_df.rdd.mapPartitions(
            lambda row_img_data_itr: ImageUtils.augment_image_generator(row_img_data_itr, no_of_augmented_images)).flatMap(lambda x: x).toDF(image_schema)

        augement_imgs_transformed_df = augement_imgs_df_new.withColumn('filenames',
                                               regexp_replace(col("image.origin"), images_input_dir, images_output_dir))
        return augement_imgs_transformed_df


    no_of_augmented_images = 5
    augement_imgs_transformed_df = image_transform_augmented(no_of_augmented_images)
    augement_imgs_transformed_df.write.format("org.apache.spark.ml.source.image.PatchedImageFileFormat")\
        .mode("overwrite").save(images_output_dir)






