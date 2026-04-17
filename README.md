## 📝 Relatório

👤 Identificação: **Nome Completo: Lucas Vinicius Santos Leonel**


### 1️⃣ Resumo da Arquitetura do Modelo

`train_model.py`.


1. Fluxo de Tratamento de Dados

  - `Carregamento do Dados:` O dataset MINIST é devidido em 60.000 imagens para realização do treinamento e 10.000 imagens para teste. 
  
  - `Normalização:` Os valores dos pixels variam de 0(Preto total) a 255(branco total) é relizado a normalização para escala (0 - 1)
  
  - `Reshaping:`As imagens são redimensionadas para o formatao (28, 28, 1), onde o 1 representa a cor (tom cinza). 

2. Arquitetura da CNN 

| Camada | Tipo | Função |
| :---: | :---: | :---: |
| `Conv2D(32)` | Concolucional | Aplica 32 filtros diferentes para detectar caracteristicas simples |
| `Maxpooling2D` | Subamostragem | Usada para compactar a imagem. Verifica um quadrado 2 X 2 de pixels e mantém o maior valor |
| `Conv2D(64)` | Concolucional | Aplica 64 filtros para detectar combinações mais complexas |
| `Maxpooling2D` | Subamostragem | Sefunda redução para evitar overfitting e reduza o processamento |
| `Flatten` | Planificação | Transforma a matriz 2D em um vetor de 1D para ser "enviado" para rede maior |
| `Dense (64)` | Totalmente Conectada | Camada intermediária de processamento com a aativação ReLU |
| `Dense (10)` | Saída | Camada final com ativação Softmar, que gera probabilidades para as 10 classes (os números 0 - 9) |

3. Estratégia de Aprendizado 
  
  - `Otimizador adam:` É utilizado para ajustar a taxa de aprendizado dinamicamente
  
  - `Loss (Perda):` Aplicado para classificação de multiplas classes onde os rótulos de cada classe é um número inteiro.  
  
  - `Métricas (Acurácia):`Metrica utilizada para acompanhar o percentual de acerto durante a realização do treinamento. 

4. Treinamento:

 - `Cinco Épocas:` O modelo utilizado para treinamento é divido em 5 épocas, onde cada época processa todo o conjunto de treinamento e valida o desenpenho com conjunto de teste. Ademais, a perfomace final do algoritmo é medida pela acurácia. 

 -`Final:` No final do processo o modelo treinado é salvo com nome `model.h5` para prosseguimento do processo. 


### 2️⃣ Bibliotecas e Tecnologias Utilizadas

- TensorFlow(v2.11.0) & Keras: Biblioteca utilizada para o desenvolvimento, treinamento e execução dos modelo da CNN. Adiconamente, foi utilizado o keras (Uma interface da biblioteca TensorFlow) para a construção de componentes importantes da CNN.

- Numpy (v1.26.4): Utilizada de forma indireta para normalização e o reshape das imagens. 

- Modulo Nativo (OS): é uma biblioteca nativa que segue a mesma versão do python 
 
- Linguagem Python (v3.11.2): base de desenvolvimento. 

- Ambiente de Desenvolvimento: VS Code (Visual Studio Code) 


### 3️⃣ Técnica de Otimização do Modelo

`optimize_model.py`.


1. Técnica Utilizada

  - `Post-Trainig Quantization:` A ténica de Quantização Pós-Treinamento, onde os valores decimais de alta precisão (Float32) são reduzidos para `int8` ou `floats` menores. Isso reduzir de forma considerável o tamanho do arquivo final. Além disso, esse técnica impacta na fforma como dispositivos de sistema embarcados processam cálculos com número inteiros impactando na velocidade de inferência.  

2. Processo de Conversão: 

  - `Instanciação do Conversor:` prepara toda aestrutura da rede para o novo formato.  

  - `Aplicação da Otimização:` Usa-se a estratégia `DEFAULT` para busca o melhor equilíbrio entre perda mínima de acurácia e ganha áximo de desenpenho, para que o modelo não perca a sua eficiência. 

  - `Serialização:` Comando `converter.convert()` gera o grafo otimizado que é scrito em um arquivo como `.tflite`


### 4️⃣ Resultados Obtidos

Apresentação de Todas as metricas de desempenho das 5 épocas de processamento implementado na CNN 

| Epoch | Acurácia (Treino) | Perda (Treino) | Acurácia (Teste) | Perda (Teste) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 95,43% | 0,1492 | 98,32% | 0,0521 |
| 2 | 98,49% | 0,0481 | 98,67% | 0,0399 |
| 3 | 98,97% | 0,0333 | 98,71% | 0,0403 |
| 4 | 99,22% | 0,0259 | 99,03% | 0,0308 |
| 5 | **99,44%** | **0,0184** | **98,99%** | **0,0296** |

**Acurácia Final:** O modelo atingiu **98.99%** de precisão nos dados de teste.  



### 5️⃣ Comentários Adicionais (Opcional)

Utilize este espaço para comentar:

- Dificuldades encontradas, aprendizados e próximos passos:  
O desafio foi particularmente interessante, pois vai ao encontro de uma das áreas que mais desperta meu interesse. Inicialmente, compreender todo o processo de implementação de uma CNN e entender as nuances do problema exigiu algumas horas de dedicação. Porém, com algunas horas de dedicação o projeto fluiu e consegui obter bons avanços.Por fim,  um dos principais aprendizados foi a base no desenvolvimento de redes neurais, área na qual pretendo me aprofundar ainda mais, mantendo o foco em minha evolução na Ciência de Dados. 

- Limitações do Moldelo:
Embora o modelo tenha se mostrado eficiência, como visto nas metricas apressentadas no tópico anterior. Um dos princiapis limitações é que o modelo foi desenvolvimento esperando imagens de compirmento 28 X 28 em tons de cinza, onde cad digito tem uma tonalidade mais escura enquanto o fundo é branco. Por conta disso, se uma imagem não estiver com essas cracteristicas isso pode impactar de forma consideravél o desenpenho do modelo. 

****
