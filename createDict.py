import pandas as pd

resultados_lpq = {}

for i in range(1, 228):
    filename = f'Resultado/Boa/histBoa{i}.txt'
    with open(filename, 'r') as f:
        resultado_lpq_laranja = f.read().strip().split('\n')
        resultado_lpq_laranja = list(map(float, resultado_lpq_laranja))
        resultados_lpq[i] = resultado_lpq_laranja

print(resultados_lpq)
df_lpq = pd.DataFrame.from_dict(resultados_lpq, orient='index')
df_lpq.to_csv('boa_lpq.csv', index=False)