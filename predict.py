import json
import numpy as np
import argparse
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

def process_image(image_path):
    im = Image.open(image_path)
    image = np.asarray(im)
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

# def predict(image_path, model, top_k):
#     image_arr = process_image(image_arr)
    
#     top_k_prediction = tf.math.top_k(prediction, k = top_k, sorted = True)
#     probs = tf.squeeze(top_k_prediction.values).numpy()
#     class_index = tf.squeeze(top_k_prediction.indices).numpy()

#     return probs, class_index

def predict():
    parser = argparse.ArgumentParser(description='Image Classifier App')
    parser.add_argument('image_path', action = 'store', type = str)
    parser.add_argument('saved_model', action = 'store', type = str)
    parser.add_argument('--top_k', action = 'store', default = 5, type = int)
    parser.add_argument('--category_names', action = 'store', default = 'label_map.json')
    
    ### Load label_category map Json file
    with open(parser.parse_args().category_names, 'r') as fname:
        label_category = json.load(fname)
    
    ### Open and process the image
    image_arr = process_image(parser.parse_args().image_path)
    
    model = tf.keras.models.load_model(parser.parse_args().saved_model, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    prediction = model.predict(np.expand_dims(image_arr, axis = 0))
    top_k_prediction = tf.math.top_k(prediction, k = parser.parse_args().top_k, sorted = True)
    probs = tf.squeeze(top_k_prediction.values).numpy()
    label = tf.squeeze(top_k_prediction.indices).numpy()
    
    print('\nPredictions for the input image {}'.format(parser.parse_args().image_path))
    for i in label:
        print(label_category[str(i+1)])

        
if __name__ == "__main__":
    predict()

    

