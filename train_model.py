import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


#importando o dataset e divindo em Treinamento e teste
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

#Normalização
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1,28,28, 1)
x_test = x_test.reshape(-1,28,28,1)

#Criando a Rede Neural 
modelo_CNN = tf.keras.Sequential([
    
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)), 
    
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(), 
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') 
]) 

#Definindo a estratégia que o modelo CNN usuára para aprender 
modelo_CNN.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
)

modelo_CNN.summary()

#Realizando o treinando do modelo
history = modelo_CNN.fit(x_train, y_train,
                       epochs=5,
                       validation_data=(x_test, y_test)
                       )

#Avaliando conjunto de teste e printando a acuracia observada no modelo  
test_loss, test_acc = modelo_CNN.evaluate(x_test, y_test, verbose=2)
print(f"Acurácia no teste: {test_acc*100:.2f}%")


"""
 ---- Codigos para plotar metricas -------
# Gerar predições
y_pred = modelo_CNN.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Plotar acurácia
plt.plot(history.history['accuracy'],     label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Época'); plt.ylabel('Acurácia')
plt.legend(); plt.show()

# Plotar loss
plt.plot(history.history['loss'],     label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Época'); plt.ylabel('Loss')
plt.legend(); plt.show()


#PLotando a Matriz de confusão
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=range(10),
            yticklabels=range(10))
plt.xlabel('Predito'); plt.ylabel('Real')
plt.show()

"""

#Salvando o modelo
modelo_CNN.save('model.h5')