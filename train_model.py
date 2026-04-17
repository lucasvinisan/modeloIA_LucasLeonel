import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets


#importando o dataset e divindo em Treinamento e teste
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

"""
x_train -> (60000, 28, 28)
x_test -> (10000, 28, 28)

(Quantidade, altura, largura)
x_train : É composto por 60000 imagens 28 X 28 (linhas e colunas) 

"""

x_train, x_test = x_train / 255.0, x_test / 255.0
"""
0 (Preto total) e 255 (Branco total)
Normalização dos valores na escala entre 0.0 e 1.0  

"""

x_train = x_train.reshape(-1,28,28, 1)
x_test = x_test.reshape(-1,28,28,1)
"""
x_trein tem o formato (60000, 28, 28)
Ao realizar x_train.replace(): 
    -> -1 : Mannter o número de exemplos (no caso 60000)
    -> 28,28 : Manter a largura e altura da imagem 
    -> 1: Definindo que a imagem contém apenas 1 canal de Cor (Tom cinza)
"""


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

modelo_CNN.save('mnist_model.h5')