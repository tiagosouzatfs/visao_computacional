# Visão Computacional
Repositório para armazenar o conteúdo do módulo de Visão Computacional do PPgTI

## Para reproduzir esses exercícios e atividades é opcional mas altamente recomendado utilizar ambiente virtuais com python para utilizar os requerimentos de bibliotecas e suas versões. Pode-se utilizar o colab, jupyter ou o vscode com a extensão do jupyter para a execução.

**Verifique o conteúdo da disciplina** [aqui!!!](https://heltonmaia.com/computervision/intro.html)

### Criando ambientes virtuais com python
Padrão Python
```
python -m venv env      # Cria
source env/bin/activate # Ativa (Linux/Mac)  |  env\Scripts\activate (Win)
deactivate              # Desativa
```

pip install virtualenv
```
virtualenv env                # Cria
source env/bin/activate       # Ativa (Linux/Mac)  |  env\Scripts\activate (Win)
deactivate                    # Desativa
```

### Criar e utilizar o arquivo de requerimentos

O comando `pip freeze` lista todos os pacotes instalados no formato **nome==versão**

### Gerar um arquivo requirements.txt com todas as dependências do ambiente
`pip freeze > requirements.txt`

### Instalar dependências a partir de um arquivo requirements.txt
`pip install -r requirements.txt`

### Instalar uma versão exata de um pacote
`pip install numpy==1.21.0`

### Listar versões disponíveis de um pacote usando PyPI
`pip index versions numpy`

### Atualizar um pacote específico
`pip install --upgrade numpy`

### Remover um pacote (exemplo: numpy)
`pip uninstall numpy -y`

**O argumento -y confirma a remoção automaticamente.**

## Se for utilizar o Colab
Os comandos `pip` e **linux** são executados usando uma exclamação na frente, ex: `!pip` ou `!mkdir`.

### Requerimentos dos códigos dos capítulos
Capítulo 1:
* Python na versão 3.10.12
* Consulte os requerimentos do capítulo 1 [aqui!!!](https://github.com/tiagosouzatfs/visao_computacional/blob/main/cap1/requirements.txt)

Capítulo 2:
* Os requerimentos e bibliotecas estão no próprio notebook

Capítulo 3:
* Os requerimentos e bibliotecas estão no próprio notebook

Capítulo 4:
* Não foram realizados exercícios

Trabalho Final:
* Utilize GPU, com CPU vai durar muito tempo
* Python na versão 3.10.12
* Consulte os requerimentos do trabalho final [aqui!!!](https://github.com/tiagosouzatfs/visao_computacional/blob/main/trabalho_final/parte_pratica/requirements.txt)