# Google Colab Tips

##

https://www.kdnuggets.com/2020/06/google-colab-deep-learning.html



Why Colab
 

There are several benefits of using Colab over using your own local machines. Some of the benefits of Colab are

You do not require to do an environment setup. It comes with important packages pre-installed and ready to use.
Provides browser-based Jupyter Notebooks.
Free GPU
Store Notebooks on Google Drive
Importing Notebooks from Github
Document code with Markdown
Load Data from Drive
and much more

## Hardware Selection (GPU or TPU)
 

Colab’s biggest advantage is that it provides free support to GPU and TPU. You can easily select GPU or TPU for your program by Runtime > Change runtime type.

## Como saber se tem CPU

torch.cuda.is_available()

e depois para inidicar para usar a cuda

````
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# indica que o classificador use cuda
regressor.to(device)

# Processamento da rede

for epoch in range(2000):
    
    ...
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
		# indica para usar cuda
        inputs, labels = inputs.to(device), labels.to(device)

	...

...

Pegando dados que estão sendo usado na GPU
# colocando dadaso para serem prcessados pela GPU
previsores = previsores.to(device)
# Aplicando a rede neural
previsoes = regressor.forward(previsores)
# tirando da gpu e pegando
previsoes.cpu().detach().numpy()
````

## Ler valores em notaçâo científica

O "e-01" indica o 10^e (no cass e = -1)

9.9229e11  = 992290000000
9.9229e2   = 992.29
1.0000e+00 = 1
9.9229e-01 = 0.99229
3.6789e-02 = 0.036789
1.0005e-06 = 0.0000010005


## Atalhos

Deletar Row: CTRL + M + D
Comment : CTRL + / 

Ver comandos
+ Bastar deixar hover nos comandos

## Verificar Configurações ...

Edit > Settings > ...
+ Ver se está usadno Python 3 e usando GPU

## Instalar pip packages

````
!pip install skorch
````


## Fazer Upload de Arquivos

A pasta raiz no FyleSystem do Colab é content, entâo, para acessar após um simples upload na raiz é 

""""
previsores = pd.read_csv('/content/entradas_breast.csv')
classe = pd.read_csv('/content/saidas_breast.csv')
""""




LINK : https://medium.com/@tuewithmorris/google-colab-notebooks-keyboard-shortcuts-aa6a008fb91b

### Acessar a MV
+ A MV é linux, você pode executar comandos linux nela pelo Colab através de !.
Exemplos:
````
!head train.csv
!apt-get install libgeos-3.5.0
!apt-get install libgeos-dev
!pip install https://github.com/matplotlib/basemap/archive/master.zip

````

### Acessar o GDrive
 Acessar o seu Drive para pegar os arquivos
````
from google.colab import drive
drive.mount('/content/drive')
````
Vai abrir um link, pelo qual você acessa com sua conta Google
Depois, você pega  um codigo_de_autorizaçâo e insere embaixo
No final,, deve ter na pasta /content/drive/ as coisas do seu drive


### ShortCuts

+ Add Code Cell : CTRL + M + B
+ Delete Cell : CTRL + M + D
+ Exec Cell: CTRL + ENTER || SHIFT + ENTER
+ Convert Code to Text : CTRL+ M + Y
+ COnvert Text to Code : CTRL + M + M

Insert code cell above	Ctrl/Cmd M A	A
Insert code cell below	Ctrl/Cmd M B	B
show keyboard shortcuts	Ctrl/Cmd M H	H
