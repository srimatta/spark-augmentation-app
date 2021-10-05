# spark-augmentation-app

run below command to generate egg file,

spark-augmentation-app % python setup.py bdist_egg 

This will generate build,SparkAugmentationApp.egg-info and dist folders. SparkAugmentationApp-1.0-py3.7.egg can be found under dist folder.

spark submit :

spark-submit --master local  --packages com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1 --py-files dist/SparkAugmentationApp-1.0-py3.7.egg app/SparkAugmentationApp.py