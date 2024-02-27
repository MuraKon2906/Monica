import tensorflow as tf 
import numpy as np



class MonicaFunc:
    #models

    #1. Sentiment analysis model
    sentiment_model=tf.saved_model.load('D:\MuraKon\Monica\sentiment_model')

    def sentimentAn(self, text):
        sentiment_model=tf.saved_model.load('D:\MuraKon\Monica\sentiment_model')

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
            print("The user seems to be happy")
            

        else:
            print("The user seems to be sad")
            

    def sentimentMod(self):
        try:
            sentiment_model=tf.saved_model.load('D:\MuraKon\Monica\sentiment_model')
            print("Sentiment model is loaded")

        except Exception as e:
            print("Error in loading the model")

    
