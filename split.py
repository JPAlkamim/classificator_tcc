#LPQ
#dividir arrays
def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

#Pegar informaçoes do txt LPQ
# file_controle = open('Features/histControle7new.txt','r')
file_tumor = open('histBoa.txt','r')


# hist_controle  = []
# for line in file_controle:
#     line = line.replace(" \n","")
#     hist_controle.append(float(line))
    
# hist_controle = split(hist_controle,256)

hist_tumor  = []
for line in file_tumor:
    line = line.replace(" \n","")
    hist_tumor.append(float(line))
    
hist_tumor = split(hist_tumor,256)
print(hist_tumor)