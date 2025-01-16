##all tansformation code
#feature eng
#data cleaning
#convert cat to numeric
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer#create the piprline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object#just to save the pickle file  
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_columns=['reading score','writing score']  
            categorical_columns=['gender',
                              'race/ethnicity',
                              'parental level of education',
                               'lunch',
                               'test preparation course']


            num_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )    
            cat_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=True)),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            ) 
            logging.info(f"numerical columns standard scaling done:{numerical_columns}") 
            logging.info(f"cat columns encoding done:{categorical_columns}")  

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_columns",cat_pipeline,categorical_columns)
                ]
            )     
            return preprocessor       
        except Exception as e:
            raise CustomException
#starting dtata ransformation
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test completed")
            logging.info("obtaining preprocessing object")
            preprocessin_obj=self.get_transformer_object()
            target_column_name="math score"
            numerical_columns=['reading score','writing score']  
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying processing on training dataframe and test dataframe")
            
            input_feature_train_array=preprocessin_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessin_obj.transform(input_feature_test_df)

            train_arr=np.c_[

                input_feature_train_array,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[

                input_feature_test_array,np.array(target_feature_test_df)
            ]

            logging.info("saved preprocessing object")

            save_object(
#saving this pickle in hard disk
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessin_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:#we write is so that we can see the error in the terminal

            raise CustomException(e,sys)
                   
            