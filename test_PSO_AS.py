import numpy as np
import random 
import matplotlib.pyplot as plt
from PSO import Particula, Enjambre
import sys
np.set_printoptions(threshold=sys.maxsize) 

#cargamos el entorno con los objetivos

entorno = np.genfromtxt('test.csv', delimiter=',')
print(entorno)

#############################
#       Análisis
#############################

grafico = np.zeros((30, 2))

for j in range(1, 31):
    enjambre = Enjambre(j, entorno)

    it = 50

    for i in range(0, it-1):
        enjambre.evaluar_enjambre()
        enjambre.actualizar_w(i,it)
        enjambre.actualizar_posiciones()

    grafico[j-1, 0] = j
    grafico[j-1, 1] = enjambre.mejor_valor_global

print(grafico)
fig, ax = plt.subplots()
ax.plot(grafico[:,0], grafico[:,1], linewidth=2)
plt.show()