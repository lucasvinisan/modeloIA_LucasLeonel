## 📝 Relatório

👤 Identificação: **Lucas Vinicius Santos Leonel**

### 1️⃣ Resumo da Arquitetura do Modelo

`train_model.py`.


1. Fluxo de Tratamento de Dados

  - `Carregamento do Dados:` O dataset MINIST é devidido em 60.000 imagens para realização do treinamento e 10.000 imagens para teste. 
  
  - `Normalização:` Os valores dos pixels variam de 0(Preto absoluto) a 255(branco absoluto). Para realização do treinamento, os valores são normalizados para o intervalo 0 e 1. 
  
  - `Reshaping:`As imagens são redimensionadas para o formato (28, 28, 1), onde 28 X 28 é a resolução em pixels e o valor 1 representa o canal de cor em escala de cinza. 

2. Arquitetura da CNN 

| Camada | Tipo | Função |
| :---: | :---: | :---: |
| `Conv2D(32)` | Convolucional | Aplica 32 filtros diferentes para detectar caracteristicas simples (Como bordas, contornos e texturas) |
| `Maxpooling2D` | Subamostragem | Usada para compactar a imagem. Verifica um quadrado 2 X 2 de pixels e mantém o maior valor |
| `Conv2D(64)` | Convolucional | Aplica 64 filtros para detectar combinações mais complexas (Ccmo curvas, formas geométricas e padrões de textuta) |
| `Maxpooling2D` | Subamostragem | Segunda redução para aumentar a eficiência computacional e controlar o overfitting |
| `Flatten` | Planificação | Transforma a matriz 2D em um vetor de 1D para entrada nas camadas maiores |
| `Dense (64)` | Totalmente Conectada | Camada de neurônio onde cada entrada se conecta a cada saída. Utiliza a ativação reLU para introduzir a não-linearidade e para aprender padrões complexos |
| `Dense (10)` | Saída | Camada final com ativação Softmax, que gera as probabilidades de cada imagem pertencer a cada uma das 10 classes (digitos de 0a 9) |

3. Estratégia de Aprendizado 
  
  - `Otimizador adam:` É utilizado para ajustar a taxa de aprendizado dinamicamente
  
  - `Loss (Perda):` Aplicado para classificação de multiplas classes onde os rótulos de cada classe é um número inteiro.  
  
  - `Métricas (Acurácia):`Metrica utilizada para acompanhar o percentual de acerto durante a realização do treinamento. 

4. Treinamento:

  - `Cinco Épocas:` O modelo utilizado para treinamento é divido em 5 épocas. Em cada época, o algoeitom processa o conjunto de dados de treinamento e valida o desempenho com conjunto de teste. A performance final é medida pela acurácia alcançada ao fim do ciclo. 

  - `Final:` No final do processo, o modelo treinado é exportado com o nome `model.h5`, permitindo a implementação dos etapas seguintes. 


### 2️⃣ Bibliotecas e Tecnologias Utilizadas

- TensorFlow(v2.11.0) & Keras: Biblioteca utilizada para o desenvolvimento, treinamento e execução dos modelo da CNN. Adiconamente, foi utilizado o keras (Uma interface da biblioteca TensorFlow) para a construção de componentes importantes da CNN.

- Numpy (v1.26.4): Utilizada de forma indireta para normalização e o reshape das imagens. 

- Modulo Nativo (OS): é uma biblioteca nativa que segue a mesma versão do python 
 
- Linguagem Python (v3.11.2): base de desenvolvimento. 

- Ambiente de Desenvolvimento: VS Code (Visual Studio Code) 


### 3️⃣ Técnica de Otimização do Modelo

`optimize_model.py`.


1. Técnica Utilizada

  - `Post-Trainig Quantization:` A técnica de Quantização Pós-Treinamento no qual os valores decimais de alta precisão (Float32) são reduzidos para formatos menores, como o `int8` ou `floats` . Esse processo reduzir de forma considerável o tamanho do arquivo final e otimiza a forma como os sistemas embarcados processam cálculos, resultando em um ganho considerável na velocidade de inferência.  

2. Processo de Conversão: 

  - `Instanciação do Conversor:` Etapa em que o modelo treinado é carregado, preparando toda a estrutura da rede para a tradução do novo formato.  

  - `Aplicação da Otimização:` Utiliza a estratégia `DEFAULT` para busca o melhor equilíbrio entre perda mínima de acurácia e ganha máximo de desempenho, para que o modelo não perca a sua eficiência mesmo após a sua otimização. 

  - `Serialização:` O comando `converter.convert()` gera o grafo otimizado, que é escrito em um arquivo do tipo `.tflite`, pronto para ser utilizado em dispositivos. 


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

- Dificuldades encontradas, aprendizados e próximos passos: 
O desafio foi particularmente interessante, pois vai ao encontro em uma das áreas de maior interesse em minha trajetória. Inicialmente, compreender todo o processo de implementação de uma CNN e as nuances do problema exigiu horas de dedicação. No entanto, as ideias foram se ajustando, o projeto fluiu e permitiu avanços consideráveis. Por fim,  um dos principais aprendizados foi consolidar a base teórica e prática de desenvolvimento de redes neurais, área na qual pretendo me aprofundar ainda mais, mantendo o foco em minha evolução na Ciência de Dados. 

- Limitações do Moldelo:
Embora o modelo tenha se mostrado eficiência, conforme as metricas apresentadas anteriormente, a CNN possui algumas limitações. O modelo foi desenvolvido para imagens de dimensões 28 X 28, onde cada digito apresenta uma tonalidade mais escura sobre um fundo branco. Por conta disso, qualquer variação nesse padrão (ruídos, inversão de cores ou resolução diferentes) pode impactar de forma consideravél o desempenho e a precisão do modelo. 

****
