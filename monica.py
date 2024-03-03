import tensorflow as tf 
import numpy as np
import os


class MonicaFunc:
    #models

    #1. Sentiment analysis model
    

    def sentimentAn(self, text):
        monica_project_folder=os.getcwd()
        sentiment_model_folder='sentiment_model'
        sentiment_model_path=os.path.join(monica_project_folder,sentiment_model_folder)
        self.print_sent_mod_path=sentiment_model_path
        
        sentiment_model=tf.saved_model.load(sentiment_model_path )

        examples=[text]
        # Assuming the model expects a   2D array where each row is a single example
        # Convert the list of examples to a   2D NumPy array
        examples_array = np.array(examples)

        # Access the model within the _UserObject
        sentiment_model = sentiment_model.signatures['serving_default']
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        # Now you can use the predict method
        # Ensure the input is correctly formatted as a TensorFlow tensor
        predictions = sentiment_model(tf.constant(examples_array))

        # Convert the predictions tensor to a NumPy array and print the value
        predictions_array = predictions['activation'].numpy()
        self.sentiment_score=predictions_array[0][0]

        if self.sentiment_score>0.5:
            return "The user seems to be happy"
            

        else:
            return "The user seems to be sad"
            

    def sentimentMod():
        try:
            monica_project_folder=os.getcwd()
            sentiment_model_folder='sentiment_model'
            sentiment_model_path=os.path.join(monica_project_folder,sentiment_model_folder)
            
            if tf.saved_model.load(sentiment_model_path ):
                return"Sentiment model is loaded"

        except Exception as e:
            return "Error in loading the model"

    
