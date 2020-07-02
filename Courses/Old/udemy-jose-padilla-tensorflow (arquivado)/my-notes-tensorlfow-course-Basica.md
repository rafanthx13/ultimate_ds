# Curso TensorFlow - Jose Padilla



## Log

- **Data de Inicio**: 24/06/2018
- **Data de Fim**: 
- **Site Udemy**: https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/learn/v4
- **O que enntedi do conteudo do curso:** 
- **Avaliação:** 



## Topicos

1. Instalaçâo de Python e TensorFlow
2. Revisão de Machine Learning
3. Passada rápida nas principais bibliotecas que serão usadas (pandas, numpy, scki-learn)
4. Introduçao a redes neurai 
   1. será bem lenta e nao usara tensor flow
5. Uso do Tensor Flow





## 1. Instalaçâo

+ é passado um aqruivo para criar um ambiente anaconda, num arquivo `yml` . Para criar esse ambiente basta fazer:
  + `conda env create -f tfdl_env.yml`
+ Para ativar, o comando Abaixo:
  + `activate tfdeeplearning` Para Windows e cmd
  + `source activate tfdeeplearning` para Linux|Mac e GitBash
+ Para desativar
  + `deactivate` Para Windows e cmd
  + `source desactivate` para gitbash
+ Para executar o jupyter notbeooj
  + `jupyter notebook`
  + Quando voce esta usando esse ambiente, vai abrir na pasta do projeto, enquanto que o normla isso nao acontece.
  + Use muito `TAB` direot do Jupyete, pois oferece a documentaçâo mais completa e instantanea



## Revisão de MachineLearning

+ ML Aprende interando sobre os dado, encontrando *insights* nos dados que até mesmo não sabiamos
+ A tres forma
  + Surpevising learing
  + Unsurpevising learning
  + Refoirciment Learning

#### **1. Supervised Learning**

+ Sinônimo de Rôtulo, onde auma row tem sua saida conhecida, e , a ML deve aprender a executar aquela row para chegar ao rótulo
+ Usado para problemas de classificação e regressão
  + Na classificaçâo, o rótulo é discreto 
    + Ex: Dado peso e altura prever se é homen ou mulher
  + na regressão o rótulo é continuo
    + Ex: Dado o número de quartos e o valor do métro quadrado, prever o preço da casa (o preço é continuo)

#### **2. Unsurpevised Learning**

+ Quando você não tem esse rótulo para os dados, então o modelo de ML deverá agrupar os dados e cada grupo será um rótulo que ele mesmo vai dar
+ Usado para Clusterização onde que os grupos devem ser interpretados para encontrar seus valore semânticos corretos. Por isso, é necessário ter conhecimento do problema para fazer isso corretamente
  + Ex: Dado Altura e Peso de um cachorro agrupar por grupos (se o cenário for bom, vai conseguir então agrupoar por raça)

#### **3. Reinforcement Learning**

+ Quando trabalha sobre tarefas complexas: jogar video games, dirigir carro, onde ocorre o aprendizado por tentativa e erro
+ Componentes
  + Agente: Decisor de ações
  + Ambiente: com o qual o agente haje
  + Ações: As ações que ele pode fazer
+ O agente escolhera as açoes para maximiza uma métricas, e, asism aprender a melhor politica para cada ambiente

#### Treinamento da Superviside Learning

1. Aquisiçâo de Dados
2. Limpagem, Formatação, Normaçização  de dados
   1. Esse passo é superimporntante, muitas vezes você vai gastar mais tempo aqui do que no modelo ML
3. Separaçâo de dados em *train* e *test*. em geral, usa-se 30%
4. Trainament do model (tensorflow ou redes neurais)
5. Testar *test* com o modelo treinado do *train* e avaliar
6. Depois de avaliado é feito reajuste do modelo
7. Por testa sobre qualquer dados

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img01.png)





#### Treinamento da Unsupervised Learning

1. Não há sepraçao de train/test

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img02.png)

#### Conjunto Hold Out

+ Serve para dar uma métrica final, a seu modelo, assim, sao dados que nunca foram vistos e que nem foram motivo de ajuste de seu modelos

  ![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img03.png)







#### Métricas para Supervised Learning - Classificação

Vamos elscoher o acruracai para ser a nétricas do nosso ML

+ Acurrária (Acurracy) : Numero de corretos divido pelo total
+ Recall 
+ Precision

+ Regression - metricas

  Sâo medidas para dizer: o quão distaten está o núermo encontrado para o que ele deveria ser o quanto de erro tem do valor que deveria ser

  + MAE (Mean Average Error)
  + MSE (Mean Squared Error)
  + RMSE (Root means Squared Error)

#### Unsupervising Learning

+ Mais difíci de avaliar e depende da tarefa
+ Podemos avaliar pela homogenidade dos clusters ou por rand index

#### Reinforcement Learning

+ é algo óbvio: o quao bom é o modelo para a tarefa a ser executada





## 2. Review

+ Numpy
+ Pandas
+ Matplotlib
+ Sckilearn

#### Numpy

+ Converte lista em array_list, estrutura prória do numpy
  + `np.array(my_list)`
+ Criar Array direto como range
  + `np.arange(0,10) => array([0 .. 9]`)
  + Sintaxe: (start, stop, step)
  + Inclui o primeiro, mas na inclui o segundo
+ Array de Zeros ou uns
  + `np.zero(3) ou np.zeros( (3,4))`
  + `np.onse(3) or np.ones((4,5)) `
  + Vai criar array de zero sendo float, sendo float, asim nao vai perder informaloes
+ Arra y de elementos separados de forma linear voce escolher o número de espaços entre um nuero inicial a outro
  + `np.linspace`
  + Exenplo: `np.linspace(0,10,5)` => 0, 2,4,6,8,10

Rando Numbers

+ random inteiro
  + `np.random.randint(low, high, size # do vetor gerado, sua dimensao)`
  + `np.random.randint(0,10) # vai gerar de 0 a 10` 
  + `np.random.randint(0,10,(3,3))` vai gerar matrix 3x3
+ Random Seed
  + Semente que permite que os núermos alertorios possam ser reproduzidos. Se voce define uma `seed` ao gerar os números random, vâo gerar os mesmos mesmo sendo de máquinas diferentes
  + 

`np.random.seed(101)`

Observe que, quando eu rodar um random, vai resetar essa semente, entao, eu devo colocala toda hora NO JUPYTER



Operaçôes

```python
np.random.seed(101) # watch video for details
arr = np.random.randint(0,100,10)
==> array([95, 11, 81, 70, 63, 87, 75,  9, 77, 40])
```

+ arg.max => retorna maior valor




```python
arr.max() # 95
```


```python
arr.min() # 4
```


```python
arr.mean() # média aritmetica = 60.799999999999997
```


```python
arr.argmin() # 7
```


```python
arr.argmax() #  0
```



+ Alterar dimensoes do arrya, será muito util
  + `np.reshape`



````python
mat = np.arange(0,100).reshape(10,10)

array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
       
       row = 0
col = 1

# acessando um elemento
mat[0,1] # 1
# acessando slices
mat[:,col] # array([ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91])

# lembrando em slice, inlcui o primeiro e vai até o ultimo que é exluido
````

Operações Bollenos como pandas

+ retorna valores bolleandos

````python
mat > 50

array([[False, False, False, False, False, False, False, False, False,
        False],
       [False, False, False, False, False, False, False, False, False,
        False],
       [False, False, False, False, False, False, False, False, False,
        False],
       [False, False, False, False, False, False, False, False, False,
        False],
       [False, False, False, False, False, False, False, False, False,
        False],
       [False,  True,  True,  True,  True,  True,  True,  True,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True],
       [ True,  True,  True,  True,  True,  True,  True,  True,  True,
         True]], dtype=bool)

mat[mat>50]

==>

array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,



````

pandas é feito como numpy





VISUALIZAÇÂO DE DADOS

+ `%matplotlib inline`

  + Sem isso, nao mostra no JUPYER, mas tambem, so funciona no Jupyer

  ````python
  plt.plot(x,y,'r--')
  plt.xlim(2,4)
  plt.ylim(10,20)
  plt.title("Zoomed")
  plt.xlabel("X Axis")
  plt.ylabel("Y Axis")
  ````

  Vai gerar

  ![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img05.png)

Criar imagens paratir de array (COnvolutional Nets)





##### SciKit-Learn

+ Kit com varisas ferrametnasde DataScien. 
+ Nos o usaremos para fazer `PREPROCESSAMENTOS o`  tratamento de dados e seperação de addos de `train` e de `test`





### Rees Neurais

+ RNA (ANN Artificial Neural Networks): se baseia no modelo biologio, chamado comunemte de perceptron

+ Como no original, temos entrada e saida:

  + Dentritos sao as INputs que serão as features
  + Axon: output

  ![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img06.png)

+ Há os pesso

  ![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img07.png)

+ Pode haver necessiade do bias para garantir uma boa saida independente do peso/entradas (pro caso de serme 0)

  Representaçâo matematica

  ![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img08.png)

Camdas

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img09.png)

+ Funçâo como a degrau não são sucetíveis à pequenos detalhes, por isso há a sigmóide

sigmoide e degrau

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img12.png)

funçâo tangente hiperbolica

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img10.png)

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img11.png)

##### Cost Functions - Avaliando o neuronio (measurement of error)

Erro quadratico

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img13.png)

Entropia

+ Ela identifica quando o erro é grande e assim dará um valor grande de erro,  dando um salto no aprendizado

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img14.png)

##### Aprendizagem - Gradient Descent

+ Gradient dscent: algoritmo otimizado para achar minimos locais. Queremos isso pois: **Queremos minimizar a funçâo de custo, encontrar qual a situaçâo que ela é a menor possível**

![](C:\Users\Rafael\Google Drive\Private Studies\Computer Advanced\udemy-jose-padilla-tensorflow\imgs\img15.png)

O backprpagation serrve paara os perceptron multi-layer. Para que se calcule o erro de cada neuronio apos o dado ser processado. Tem esse nome pois vai da daisda par ao começo



