import tensorflow as tf   
import numpy as np

try:
    sentiment_model = tf.saved_model.load('D:\MuraKon\Monica\sentiment_model')
    print("I am so happy")

except Exception as e:
    print("error")    

examples = [
  "As she entered the room, the enigmatic smile on his face left her unsure whether he was genuinely pleased to see her or hiding something beneath the surface.",

]

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
score=predictions_array[0][0]

if score>0.59:
    print("The user seems to be happy")
    print(score)

else:
    print("The user seems to be sad")
    print(score)
