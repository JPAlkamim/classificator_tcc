import matplotlib.pyplot as plt
import statistics

with open('histBoa.txt') as f:
    dados = [float(line.strip()) for line in f]
    stdev = statistics.stdev(dados) 
    threshold = 3 * stdev
    clean_dados = [x for x in dados if abs(x - statistics.mean(dados)) < threshold]

plt.hist(clean_dados, bins=50)
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.title('Histograma')
plt.show()