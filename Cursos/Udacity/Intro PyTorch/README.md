# Introdução a pytoch

## Índice

[TOC]

## Introdução ao Curso

Link do Curso: https://www.udacity.com/course/deep-learning-pytorch--ud188

Link do GIt: https://github.com/udacity/deep-learning-v2-pytorch

Partes do Curso

1. Intro a Redes Neurais, Perceptron, Descida do gradiente, Back Propagation
2. Entrevista com Criador do PyToch



## Introduction to a Neural Network

![1565482881643](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565482881643.png)

Como representar um Perceptron

+ Ele é um grafo, em que, a entrada, que consite de `x1` a `xn`, há nós que ligam a um nó central, e entre esses mós há pesos (w1 a wn)
+ Há tambem um nó bias/viés, que tem valor fixo e cujo peso muda. Ele tem uma valor constante e serve para mais para ser regulado pelo peso
+ É feito um somatório e produtório das entradas com os pesos.
+ Em seguida, passa por uma `step_function` que retorna a label: 0, 1



+ Perceba que, por o peso sempre acompanhar a entrada, eles sempre tem a mesma quantidade: n input, n pesos, então, podemos represneetar como matrizes:
  + Input: x1 , xn tem a dimensao (1xn) : 
  
    + ```
      [x1,x2,xn]
      ```
  
  + Weights: w1, .. wn dimesnsao de (nX1):
    
    + ```
      [w1,w2,w3]
      ```
  
+ Assim, a multiplicação dessa mastrizes da o que queremos: 

  + ```
    x1*w1 + x2*w2 até xn*wn
    ```

    

![1565482593803](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565482593803.png)

### Por que redes neurais?

![1565483355138](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565483355138.png)

A semelhança com a biologia do cérebro humano:

+ As entradas sao os dentritos

+ O nucleo é o somatório de tudo

+ E passa sinal ou nao no axon (0/1)

### Algoritmo do perceptron

![1565485602164](/home/rhavel/.config/Typora/typora-user-images/1565485602164.png)

![1565485548726](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565485548726.png)

Código

````python
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Para cada peso/vetor (indexce de 0 até len(X)
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b) # Prezis: Passando uma linha de dados X[i], os pesos e o bias
        if y[i]-y_hat == 1: # se ele é positivo (1) e classificou como negativo (0) então aumenta
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1: # se ele é negativo (0) e classificou como positivo (1) entâo diminui
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
````

### Error Fucntion

Informa o quão longe estou da solução: a cada passo eu a recalculo para saber a direção que eu vou. Em geral é a quantidade de erros da nossa previsao



Gradiente Descent

Para aplicalo corretamente a Error FUnction deve ser

**Continua e Diferenciavel**

Isso quer dizer tambem que nossa previsão deverá ser contínua. Em vez de dar uma label para cada classificaçâo, vamos dar uma valor continuo (probabliidade) para cada. Se a prob for maior que 50%, será 1, do contrtrário, rotulado de zero.



![1565636981540](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565636981540.png)

Assim as previsões serão rotaladas da forma a seguir

![1565637024995](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565637024995.png)

Assim o perceptron fica

![1565637137400](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565637137400.png)



### Função SoftMax para classificar em n classes

![1565638247115](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565638247115.png)

### One-Hot Encodig

Criar n features composotas de zero e ums, sendo que soemnte uma delas terão um bit como 1, representando a clase

![1565638130072](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565638130072.png)

### Maximun Likelihoof (máxxima verrosimilnaça)

![1565638673429](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565638673429.png)

Como classificar dois modelos? Se você observar esses dois modelos acima, perceberá que o da direita é melhor que a esquerda pois acerta mais. Uma forma de provar isso de maneira matématica é fazer o **Maximun Likelihood** : Vamos fazer um produtório das probabilidades resultantes da aplicação da sigmoied para os rótulo que esperamos: a prova de um vermelho dar vermelho e a prob. de um ponto zaul receber o rótulo de azul. 

Como na imagem acima, **O melhor modelo é o que tem maior maximun likelihhod**

Assim, calcular a probabiblidade de dar o que se quer usando nosso modelo indica o quanto de acurácia ele tem

### Cross Entropy

![1565639347001](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565639347001.png)



![1565639465338](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565639465338.png)

obs: é o negativo de tudo isso por o lod de algo entre [0,1] da um valor negativo

**Quando menotr o valor de H, melhor.**

**A entropia cruzada descreve a perda entre duas distribuições de probabilidade.**

Assim, podemos concluir que a cross entropie pode nos dizer se um modelo é bom ou ruim e comparara com outros modelos.

Qanto maior a entropia, quer dizer que fogem do caso comun (acertar) ou seja, menor a probabilidade de acontecer.

### Voltando a error function agora usando cross entropy

![1565641244649](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565641244649.png)

Sendo para y a bola foi marcada como azul ou y_chapéu como probabilidade dada pelo modelo da bola ser  azul (quanto maior, quer dizer que diz com mais certeza que é azul, e , conseguqentemente, seu inverso é a probabailidade de a ter marcado como vermelho)

+ y = 1 => marcou como azul
  + Será negativo da: probabilidade de dar azul
+ y = 0 => marcou como vermelho
  + Será o negativo da probabilidade de dar vermlho

Essa formula cancela o termo que nâo ocorreu, entao, nao precisa fazer nenhum `if`

### Gradient Descent

Igual a aquilo que vi com rita, da parte da derivada. A derivada

o  que é a derivada

Derivada quer dizer a taxa de variação: o quanto está mudando. QUanto maior, quer dizer que a função nesse ponto está crescendo/descendo muito, como a exponencial. Quanto menor, ela está mudando pouco, como uma reta.

derivada parcial: é isso que é o grandiente

derivada percial é quando uma funçâo tem várias variáveis e escolhemos uma e finjimos que todas as outras sâo constantens e fazemos sua derivada conseiderando comente uma unica variavel.

A interpretação do gradiente é a direção na qual a variação da função F é máxima.

As derivadas pariciais indicam para onde está crescento tambem, entao, o seu inverso, é para onde queremos que desça. (nosso objetivo, ter o erro o menor possivel)

![1565642853437](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565642853437.png)

So, a small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.

**Quanto menor o gradiente, quer dizer que está bem perto de chegar ao ponto final, ao mínimo global. Quanto mairo,q uer dizer que está mais distante**

![1565642703983](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565642703983.png)

![1565642797324](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565642797324.png)

Então, para cada peso, agente pega esse gradiente, inevrter e adiciona ao peso para descer a montanha.

### Algoritmo do Perceptron Final

Só um neuronio, uma única camada de processamento que gera uma reta

Here is your turn to shine. Implement the following formulas, as explained in the text.

- Sigmoid activation function

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

````python
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
````



- Output (prediction) formula

$$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$

````python
def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)
````



- Error function

$$Error(y, \hat{y}) = - y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$

````python
def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)
````



- The function that updates the weights

$$ w_i \longrightarrow w_i + \alpha (y - \hat{y}) x_i$$

$$ b \longrightarrow b + \alpha (y - \hat{y})$$

````python
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias
````

TREINAR

````python
np.random.seed(44)

epochs = 100
learnrate = 0.01

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):
            output = output_formula(x, weights, bias)
            error = error_formula(y, output)
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
````

Quando você realiza a mudança de peso há duas situações:

+ Se acertou o ponto, ele dirá "afaste a linha de mim, para não errar"
+ Se errou o ponto, ele vai fala "Chegue mais perto para acertar"

### Redes Neurais: Perceptron de Múltiplas camadas

![1565892768563](/home/rhavel/.config/Typora/typora-user-images/1565892768563.png)

![1565893141340](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565893141340.png)

![1565893329336](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565893329336.png)

Podemos combinar várias redes neurias, tornando assim de um modelo linear (que separa em uma linha) e um que separa cada vez mais de formas mais complexas. Assim, a saida de uma mini rede neural entra na próxima com um peso w, havendo assim várias sigmoides.

Uma deep network é uma rrede que tem muitas hidden layes, com muitos nós.

### Treinar uma Rede Neural

#### FedFoward (alimentação direta)

![1565893674553](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/1565893674553.png)

#### Separando os dados

Para treinar uma rede, separamos os dados entre dados de treino e dados de teste.

Geralmente é 70% ou 80%  para treinar a rede e o restante (30 ou 20%) para testá-la após o treinamente. 

#### Overfitting and Underfitting

Underfitting (Pouco Ajustado: Quando você não é especifico o suficiente na classificação dos dados

+ Comete erros de mais

Overfitting (Super Ajustado): Quando você é específico de mais no treinamento dos dados, assim fica enviezado por se ajustar a seus dados, sendo que tem que resolver o problema para qualquer instância. Assim, ele até pode acertar boa parte do seus dados de treinamento, mas poderá ser difícil acertar aoutros casos que fogem do padrão do treinamento

+ Tende a não cometer nenhum erro por ser específico demais ao dados
+ Desse conceitos, o que tem mais chance de ocorrer é o overfitting :
  + Ocorre se treinar o mesmo dados várias e várias vezes

![](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/img01.png)

![](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/img02.png)

**O que tornar um modelo Overfitting ou Underfitting é o número de epoch : ou seja, a quantidade de vezes que se executa o treinamento com os dados**

Então: **QUAL O NÚMERO IDEAL DE EPOCH QUE SATISFAZ O MELHOR CENÁRIO**

<img src="/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/img03.png" style="zoom:75%;" />

Pois sabemos que, se deixarmos muito overffitng, vai errar demais os casos de teste (os mais genéricos)

![](/home/rhavel/Cursos/Udacity/Intro PyTorch/md_images/img04.png)

Para encontrar o melhor número de Epoch é atravez da descida do gradiente, através do conceito de *early stopping* (parada precoce)



