
from app import ImageUtils
import pyspark
from app.autoaugment import CIFAR10Policy
from pyspark.sql.types import *
import os


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

    images_input_dir  = "images"
    images_output_dir = "images_processed"

    images_df = spark.read.format("image").load(images_input_dir)


    def image_transform_augmented(images_df, no_of_augmented_images, image_schema, images_input_dir, images_output_dir):

        img_transformer = ImageTransformer().setOutputCol("out")

        transformed_images = img_transformer.transform(images_df).select("out").withColumnRenamed("out", "image")
        transformed_df = transformed_images.select("image.height", "image.width", "image.nChannels", "image.mode", "image.origin", "image.data",
                                  "image.origin")

        augement_imgs_df_new = transformed_df.rdd.mapPartitions(
            lambda row_img_data_itr: ImageUtils.augment_image_generator(row_img_data_itr, no_of_augmented_images, CIFAR10Policy())).flatMap(lambda x: x).toDF(image_schema)

        augement_imgs_transformed_df = augement_imgs_df_new.withColumn('filenames',
                                               regexp_replace(col("image.origin"), images_input_dir, images_output_dir))
        return augement_imgs_transformed_df


    no_of_augmented_images = 6
    augement_imgs_transformed_df = image_transform_augmented(images_df, no_of_augmented_images, image_schema, images_input_dir, images_output_dir)
    augement_imgs_transformed_df.write.format("org.apache.spark.ml.source.image.PatchedImageFileFormat")\
        .mode("overwrite").save(images_output_dir)






