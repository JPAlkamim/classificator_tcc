# Classificado de Laranjas para Trabalhado de Conclusão de Curso

Todos os códigos utilizados para a obtenção dos resultados

## Biblioteca necessária

```bash
pip install -U scikit-learn
```

## Pastas

Todas as pastas de cada classificador se encontram com seus respectivos nomes: KNN e SVM.
Dentro da pasta do LPQ se encontra o algoritmo em MATLAB que extraiu todas as características das laranjas.


## Uso

O uso é simples, basta escrever python junto com o nome do arquivo. 

```python
python ./nome_do_arquivo.py
```

Lembrando que para testar diferentes tipos de banco basta mudar o caminho da variável

```python
X3 = np.load('Banco/X3.npy')
y3 = np.load('Banco/y3.npy')

#ou
X9 = np.load('Banco/X9.npy')
y9 = np.load('Banco/y9.npy')
```
