#submission.py
#COMP_9313_Project_2
#Author: Anant Krishna Mahale 
#zID: z5277610

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.sql import DataFrame
from pyspark.ml.classification import  LinearSVC, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F

class Selector(Transformer):
    def __init__(self, outputCols=['features', 'label']):
        self.outputCols = outputCols

    def _transform(self, df: DataFrame) -> DataFrame:
        return df.select(*self.outputCols)  

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    # white space expression tokenizer
    word_tokenizer = Tokenizer(inputCol=input_descript_col, outputCol="words")

    # bag of words count
    count_vectors = CountVectorizer(inputCol="words", outputCol=output_feature_col)

    # label indexer
    label_maker = StringIndexer(inputCol=input_category_col, outputCol=output_label_col)

    selector = Selector(outputCols=['id','features', 'label'])

    # build the pipeline
    pipeline = Pipeline(stages=[word_tokenizer, count_vectors, label_maker, selector])

    return pipeline


def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    range_value = training_df.agg({'group':'max'}).collect()[0][0]
    range_value = range_value + 1

    for i in range(range_value):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()
            
        nb_model_0 = nb_0.fit(c_train)
        nb_pred_0 = nb_model_0.transform(c_test)

        nb_model_1 = nb_1.fit(c_train)
        nb_pred_1 = nb_model_1.transform(c_test)

        nb_model_2 = nb_2.fit(c_train)
        nb_pred_2 = nb_model_2.transform(c_test)

        svm_model_0 = svm_0.fit(c_train)
        svm_pred_0 = svm_model_0.transform(c_test)

        svm_model_1 = svm_1.fit(c_train)
        svm_pred_1 = svm_model_1.transform(c_test)

        svm_model_2 = svm_2.fit(c_train)
        svm_pred_2 = svm_model_2.transform(c_test)

        if (i<1):
            temp_df = c_test.join(nb_pred_0, on = ['id']).select(c_test["*"],nb_pred_0["nb_pred_0"])
            temp_df = temp_df.join(nb_pred_1, on = ['id']).select(temp_df["*"],nb_pred_1["nb_pred_1"])
            temp_df = temp_df.join(nb_pred_2, on = ['id']).select(temp_df["*"],nb_pred_2["nb_pred_2"])

            temp_df = temp_df.join(svm_pred_0, on = ['id']).select(temp_df["*"],svm_pred_0["svm_pred_0"])
            temp_df = temp_df.join(svm_pred_1, on = ['id']).select(temp_df["*"],svm_pred_1["svm_pred_1"])
            temp_df = temp_df.join(svm_pred_2, on = ['id']).select(temp_df["*"],svm_pred_2["svm_pred_2"])
            result_df = temp_df
        else:
            temp_df = c_test.join(nb_pred_0, on = ['id']).select(c_test["*"],nb_pred_0["nb_pred_0"])
            temp_df = temp_df.join(nb_pred_1, on = ['id']).select(temp_df["*"],nb_pred_1["nb_pred_1"])
            temp_df = temp_df.join(nb_pred_2, on = ['id']).select(temp_df["*"],nb_pred_2["nb_pred_2"])

            temp_df = temp_df.join(svm_pred_0, on = ['id']).select(temp_df["*"],svm_pred_0["svm_pred_0"])
            temp_df = temp_df.join(svm_pred_1, on = ['id']).select(temp_df["*"],svm_pred_1["svm_pred_1"])
            temp_df = temp_df.join(svm_pred_2, on = ['id']).select(temp_df["*"],svm_pred_2["svm_pred_2"])
            result_df = result_df.union(temp_df)

    result_df = result_df.orderBy('id', ascending=True)
    result_df = result_df.withColumn("joint_pred_0",2*result_df.nb_pred_0 +result_df.svm_pred_0)
    result_df = result_df.withColumn("joint_pred_1",2*result_df.nb_pred_1 +result_df.svm_pred_1)
    result_df = result_df.withColumn("joint_pred_2",2*result_df.nb_pred_2 +result_df.svm_pred_2)
    result_df = result_df.select(result_df.id,result_df.group,result_df.features,result_df.label,result_df.label_0,result_df.label_1, result_df.label_2,result_df.nb_pred_0,result_df.nb_pred_1,result_df.nb_pred_2, result_df.svm_pred_0, result_df.svm_pred_1, result_df.svm_pred_2, result_df.joint_pred_0, result_df.joint_pred_1, result_df.joint_pred_2)   
    return result_df

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    temp_result_0 = base_features_pipeline_model.transform(test_df)
    temp_result_1 = gen_base_pred_pipeline_model.transform(temp_result_0)

    #find the joint probability or generate meta-parameters.
    temp_result_1 = temp_result_1.withColumn("joint_pred_0",2*temp_result_1.nb_pred_0 +temp_result_1.svm_pred_0)
    temp_result_1 = temp_result_1.withColumn("joint_pred_1",2*temp_result_1.nb_pred_1 +temp_result_1.svm_pred_1)
    temp_result_1 = temp_result_1.withColumn("joint_pred_2",2*temp_result_1.nb_pred_2 +temp_result_1.svm_pred_2)  

    temp_result_2 = gen_meta_feature_pipeline_model.transform(temp_result_1)
    temp_final = meta_classifier.transform(temp_result_2)
    final_result = temp_final.select(temp_final.id, temp_final.label, temp_final.final_prediction)
    return final_result
