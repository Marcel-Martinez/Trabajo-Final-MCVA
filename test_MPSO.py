import numpy as np
import random 
import matplotlib.pyplot as plt
from MPSO import Particula, Enjambre
import sys
np.set_printoptions(threshold=sys.maxsize)

# cargamos el entorno con los objetivos
entorno = np.genfromtxt('test.csv', delimiter=',')

print(entorno)
#############################
#       Ejecución del MPSO
#############################

#inicializamos el enjambre
enjambre = Enjambre(20, 4,entorno)

it = 50

for i in range(0, it-1):
    enjambre.evaluar_enjambre()
    enjambre.actualizar_w(i,it)
    enjambre.actualizar_posiciones()

    res = enjambre.gbest
    fo = enjambre.mejor_valor_global

    print(res)
    print(fo)
enjambre.generar_grafico()