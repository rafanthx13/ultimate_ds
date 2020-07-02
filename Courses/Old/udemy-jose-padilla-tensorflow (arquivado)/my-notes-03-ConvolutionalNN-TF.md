## Udemy - José Padilla - III Parte - CNN



# Convolutional Neral Net (CNN)

outros links 

+ https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1

## Antes de Continuar, Vamos fazer algumas revisões

Topicos

**Inicialização dos pesos**

+ Zeros: 
  + Não é uma boa escolha  pois nâo é randomico
+ Random Distribution Near Zero [0,1] 
  + Não é ótimo
  + Pode não ser bom para certas funções de ativação. Alem disso, talvez o peso seja um valor muito maior que o modulo de[0,1]
+ Outra opção: Xavier (Glorot)
  + Distribuiçâo Uniform / Normal. Chega a uma foórmula, (depois procurar mais, há o link no vídeos)

**Gradiente Descent**

Learning Rate:

## Bath Size (Definido agora)

Permite fazer o stochastic Gradiente Descent.

**RESUMINDO:** Como vimos, fazer a atualizaçâo de pesos em uma DNN é complicado, o backpropagation é muito custoso. Em vez de fazermos para cada dado passado, **podemos fazer a atualização dos pesos após n dados**. Esse **n** é o **bath_size**.

**Como funciona:** Voce passa n dados e pela rede. Para cada saida você calcula o erro. O pulo do gato é: Crie um acumulador de erro e vai acumulando os erros. Dpeois dos n dados, faz a média, e **só aí você atualiza o peso.**

**Implicações**: Assim, o passo mais custoso (backpropagation) é executado teto(M/n), onde M é o tamanho de amostras e n o bath_size. Isos reduz o número de backpropagation executados em uma grande quantidade.



### Stochastic Gradient Descent em Aprendizado de Máquina

A Descida do Gradiente pode ser lenta para executar em conjuntos de dados muito grandes. Como uma iteração do algoritmo de descida do gradiente requer uma previsão para cada instância no conjunto de dados de treinamento, pode demorar muito quando você tem muitos milhões de instâncias.

Em situações em que você possui grandes quantidades de dados, você pode usar uma variação da descida do gradiente chamada **Stochastic Gradient Descent.**

Nesta variação, o procedimento de descida do gradiente descrito acima é executado, mas **a atualização para os coeficientes é realizada para cada instância de treinamento**, em vez do final do lote de instâncias.

+ Para otimizar a performance, o que se faz é passar pela rede múltiplas amostras (por exemplo 128 amostras), calcular o erro médio delas e então realizar o backpropagation e a atualização dos pesos. Do ponto de vista da atualização dos pesos, 1 amostra = 128 amostras. Esse é um conceito mais novo, necessário principalmente no treinamento de grandes modelos de Deep Learning. 

O primeiro passo do procedimento exige que a ordem do conjunto de dados de treinamento seja randomizada. Isto é, misturar a ordem que as atualizações são feitas para os coeficientes. Como os coeficientes são atualizados após cada instância de treinamento, as atualizações serão barulhentas saltando por todo o lado, e assim o custo correspondente funcionará. Ao misturar a ordem para as atualizações dos coeficientes, ela aproveita essa caminhada aleatória e evita que ela fique “distraída” ou presa.

O procedimento de atualização para os coeficientes é o mesmo que o anterior, exceto que o custo não é somado em todos os padrões de treinamento, mas sim calculado para um padrão de treinamento.

A aprendizagem pode ser muito mais rápida com descida de gradiente estocástica para conjuntos de dados de treinamento muito grandes e muitas vezes você só precisa de um pequeno número de passagens através do conjunto de dados para alcançar um conjunto de coeficientes bom o suficiente.



**Second-Order Behavior of Gradiente Descent**

+ No primeiro dado, com pesos random, sabemos que a probabildade de erro é alta. E que, no final, a prob de erro é baixa.
+ Sabendo disso, seria interresante se pude-se aumentar o learning rate no inicio e diminuir ele no fim: Há modos chamdaos (AdaGrad, RMSProp e Adam)

Método Adam

+ Como dito, permite dar largos passos no início e menores no fim de forma automática.

Unstable / Vanishi Gradients

**O que é (Cenário):** imagine que tem uma DNN com várias camdas. A medida que você faz o Gradiente Descetnt, o valor deerro vai cada vez mais diminuindo. Em suma, na 1° camada de entrada, seus pesos vão ser muito pouco modificados, sendo que é a camada que mais influencia, sendo a 1°.

**Como melhorar isso**: Boa inicializaçâo e Normalização

Será mais discutito mais nas Recorrent Neural Net



**Overfitting vs Underfitting**

Underfitting: Te rum grande eror por pegar so uma parte dos dados

Overfitting: é quando o treinamento é tendecioso. Ocorrerá um grande erro quando for colocar dados de teste. (mais perigoso que o outro). 

+ L1/L2 Regularizador:
  +  Qunaod há muitos parametros, pode ocorrer o Overfitting, para corrigir isso usa essa estratégia (Penalisar grandes dados muito grandes)
+ Dropout
  + Remover alguns dados de treinamento durante o treinamento
+ Expandign Data:
  + Adicionar ruido aos dados



## MNIST Data

+ DataSet famoso para ML de **IMAGENS** com imange s para treino, test e validação.
+ Aquele usado por Douglas: Contem números de 0 a 9. E nos represetaremos essa imagem num array  em gray-scale (28x28) ou podemos tambem achatar (flattening) para (781, 1)
+ Para nosso projeot, o rótulo será um array de 0/1. Exmeplo: 4 => [0,0,0,0,1,000,0,0,0,0]. Essa represetnação é chamada de **hot-encoding**

**SoftMaxRegression**

+ Aquilo que fizemos com o titanic, de classificar em morreu ou não, é chamado de Regressão Logística e serve para classificações binárias

+ Agora, temos que usar esse pois há várias classes, por isso usamos o SoftMax com a  mesma  sigmoide. Só que, usando, vamos retornar

  + UMA LISTA de cada probablidade

  + Usando a abordagem parad^roa, obtivemos uma precisâo de 92%. Apesar de ser alta, bons modelos consegue 99%.

    Para conquistar essa marca, vamos usar CNN

## CNN - Teoria

+ Vamos resolver o MNIST com uma CNN chegando perto à 100%
+ Como os perceptrons, sua origem veio da biologia, do estudo de Hubel e Wiesel no cortex visual do mamíferos que ganharam premio nobel em 1981

Nossa visâo biológica

+ Em resumo, cada neuronio visual olha para uma pequena parte
+ Isso revela a seguinte ideia: "Olhar porçoes pequenas da imagem"
+ Isso inpirou Yann LeCun em 1998 a escrever um papae e hoje ele é o Director of AI research of Facebook e criou o LeNet-5 , uma arquitetura para calssificar o MNIST pela primeira vez

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img18.png)

A parte de classificaçâo é a mesma que vimos antes, uma Rede Neural com SoftMax para dar uma classe



#### Modelar uma imagem para a CNN : Tensors

+ Tensor é usar muitas matrizes dentro de outras matrizes.
+ Faremos isso e tambem colocaremos a imagem como uma quadrupla (I,H,W,C) (Image, Height, Width, Color)

##### Diferença da DNN e CNN

A DNN (Dense) todos os nós sâo conectados a todos da camda seguinte

Isso nao ocorre na CNN, pois, para se obeservar melhor a imagem, a medida que passa as camadas, os nos de uma pra seguinte ficam sendo menores. 

+ Isso ocorre pois como dito por Jordan Peterson (12 Regras para o Caos) **A VISÃO É O SENTIDO MAIS CUSTOSO** por isso, ela nâo é perfeita e tanto o cérebro quanto nós mesmo na computaçâo temos que pegar apenas pedaços da imagem,
+ Outra coisa, nessa abordagem, um pixel perto de outro sera mais perceptivel

CNN

+ Cada camada da CNN vê uma parte cada vez mair/menor da imagem

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img19.png)

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img20.png)

Na CNN, os pesos servem como filtros para analise de features. 

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img21.png)

**COM MULTIPLOS FILTROS, VOCÊ A NALISA A IMAGEM DE VARIOS ANGULOS**



POOLING OR SUBSAMPLING

+ Serve para reduzir o tamanaho da imamgem durante o processamento. é fazer uma média a cada n pixels e só passar para frente o maior/menor deles
+ Isso remove muita informaçâo

DROUP OUT

+ é colocar ruido de proposito para evitar está over-fitting

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img23.png)