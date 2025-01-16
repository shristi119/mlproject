##will have all the code for reading the data
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#dataingestion class congig:dataingestion knows where we have to save train,test,raw data 
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')#this artifact is a folder
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()#ingestin_config variable now has all the path from DataIngestionconfig

     #if data is stored in some databases so in below function we will write the code to read from the databses
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or components")
        try:
            df = pd.read_csv(r"C:\Users\sumit.DESKTOP-5DLLNM0\Downloads\mlproject\StudentsPerformance.csv")  # here we can read the dataset from mongodb or ui 
            logging.info("Read the dataset as dataframe")    #its important to keep on writing the log so whereevr exception occurs we will get to know
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train tset split has started")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)#everything is getting stored in artifacts folder
            logging.info("Ingestion completed")

            return(

                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            

#initiating and running
if __name__=="__main__" :

    obj=DataIngestion()
    obj.initiate_data_ingestion()



            