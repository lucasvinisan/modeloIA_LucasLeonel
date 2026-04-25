import tensorflow as tf
import os


#Carregando o modelo CNN criado 
model = tf.keras.models.load_model('model.h5')

#Conversor
converter_model = tf.lite.TFLiteConverter.from_keras_model(model)

#Otimização 
converter_model.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_modelo = converter_model.convert()

#Salvando os arquivos 
with open('model.tflite', 'wb') as f:
    f.write(tflite_modelo)

print('Sucesso conversão')




