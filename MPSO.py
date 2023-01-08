import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#definicion de las clases particula y enjambre


class Particula:

    def __init__(self,pos, v, grupo):
        self.pos = pos # np array[x, y]
        self.v = v
        self.grupo = grupo
        self.pbest = np.zeros(2) #mejor posición
        self.mejor_valor = 0
        self.fitness = None 
        self.registro = []
        self.it = 0

    def asignar_grupo(self, i_grupo):
        self.grupo = i_grupo
    
    def get_grupo(self):
        return self.grupo

    #Local Search, se capturan los pixeles objetivo que la partícula tenga en un radio de 10 pixceles
    def LS(self,entorno):
        pij = 0 # cantidad de pixeles con algún objetivo en él

        for i in range(1, 10):
            if(self.pos[0]+i < len(entorno)):
                pij = pij + entorno[self.pos[0]+i, self.pos[1]]

        for i in range(1, 10):
            if(self.pos[0]-i > 0):
                pij = pij + entorno[self.pos[0]-i, self.pos[1]]

        for i in range(1, 10):
            if(self.pos[1]+i < len(entorno)):
                pij = pij + entorno[self.pos[0], self.pos[1]+i]

        for i in range(1, 10):
            if(self.pos[1]-i > 0):
                pij = pij + entorno[self.pos[0], self.pos[1]-i]    

        for i in range(1, 10):
            if(self.pos[0]+i < len(entorno) and self.pos[1]+i < len(entorno)):
                pij = pij + entorno[self.pos[0]+i, self.pos[1]+i]

        for i in range(1, 10):
            if(self.pos[0]+i < len(entorno) and self.pos[1]-i >0):
                pij = pij + entorno[self.pos[0]+i, self.pos[1]-i]

        for i in range(1, 10):
            if(self.pos[0]-i > 0 and self.pos[1]+i < len(entorno)):
                pij = pij + entorno[self.pos[0]-i, self.pos[1]+i]

        for i in range(1, 10):
            if(self.pos[0]-i >0 and self.pos[1]+i < len(entorno)):
                pij = pij + entorno[self.pos[0]-i, self.pos[1]-i]            

        return pij


    #evaluar la particula según la función fitness
    def evaluar_particula(self, entorno):
        self.fitness = self.LS(entorno)/400

        if(self.fitness > self.mejor_valor):
            self.mejor_valor = self.fitness
            self.pbest = self.pos
        

    #actualizar posición de la partícula
    def actualizar_pos(self, gbest, w, c1, c2, entorno):
        #actualizamos la v
        self.v = w*self.v + c1*random.random()*(self.pbest - self.pos) + c2*random.random()*(gbest- self.pos)
        #actualizamos la posición
        self.pos = self.v.astype(int) + self.pos

        if(self.pos[0] > len(entorno)-1 and self.pos[1]>len(entorno)-1):
            self.pos[0] = len(entorno)-2
            self.pos[1] = len(entorno)-2
            self.v = 0
        elif(self.pos[0]>len(entorno)-1):
            self.pos[0] = len(entorno)-2
            self.v = 0
        elif(self.pos[1]>len(entorno)-1):
            self.pos[1] = len(entorno)-2
            self.v = 0
        elif(self.pos[0]<0 and self.pos[1]<0):
            self.pos[0] = 0
            self.pos[1] = 0
            self.v = 0
        elif(self.pos[0]<0):
            self.pos[0]=0
            self.v=0
        elif(self.pos[1]<0):
            self.pos[1]=0
            self.v=0

        #esta ultima serie de condicionales son para evitar que la partícula se salga de los limites del entorno

    def insertar_registro(self):
        self.registro.append([self.it, self.pos[0], self.pos[1]])
        self.it = self.it + 1
    
    def obtener_registro(self):
        return self.registro

        

class Enjambre:

    def __init__(self, n, n_grupos, entorno):
        self.n = n # numero de partículas
        self.n_grupos= n_grupos
        self.gbest = np.array([random.randint(0, len(entorno)-1), 
                                random.randint(0, len(entorno)-1)])
        self.mejor_valor_global = 0
        self.entorno = entorno
        self.c1 = 0.5
        self.c2 = 0.5
        self.w = 0
        self.wmax = 0.9
        self.wmin = 0.4
        self.particulas = [] #array de partículas
        self.bestGrupo = np.zeros(n_grupos)

        #creaccion del enjambre de partículas(con asignacion de grupos en este caso):
        i_grupo = 0
        count = 0
        for i in range(0, self.n-1):
            grupos = int(self.n/self.n_grupos) #asignación de grupos a cada partícula declarada

            if(count > grupos):
                count = 0
                i_grupo +=1

            self.particulas.append(Particula(pos=np.array([random.randint(0, len(entorno)-1), 
                                                        random.randint(0, len(entorno)-1)]),
                                                        v=random.randint(1, len(entorno)/2), grupo=i_grupo))
                                                        
            count += 1



    def evaluar_enjambre(self):
        grupos = round(self.n/self.n_grupos) #numero de partículas por grupo
        i_grupo=0
        count=0
        self.bestGrupo = np.zeros(self.n_grupos)
        bestPosGrupo = np.zeros((self.n_grupos, 2))

        for i in range(0, self.n-1):
            if(count > grupos):
                count = 0
                i_grupo +=1
            self.particulas[i].evaluar_particula(self.entorno)
            self.particulas[i].asignar_grupo(i_grupo)
            #se guarda el mejor valor de cada grupo creado
            if(self.particulas[i].mejor_valor > self.bestGrupo[i_grupo]):
                self.bestGrupo[i_grupo] = self.particulas[i].mejor_valor
                bestPosGrupo[i_grupo] = self.particulas[i].pbest
            count +=1
        
        #guardamos el gbest (mejor posicion global) y el mejor valor global comparando cada grupo
        for idx, i in enumerate(self.bestGrupo):
            if(i > self.mejor_valor_global):
                self.mejor_valor_global = i #guardamos mejor valor global
                self.gbest = bestPosGrupo[idx] #guardamos mejor posición

    def actualizar_w(self, t, tmax):
        # t: iteración obtenida del contador, tmax: numero máximo de iteraciones
        self.w = ((self.wmax -  self.wmin)*(tmax-t)/tmax) + self.wmin
        

    def actualizar_posiciones(self):
        for i in range(0, self.n-1):
            self.particulas[i].actualizar_pos(self.bestGrupo[self.particulas[i].get_grupo()], self.w, self.c1, self.c2, self.entorno)
            self.particulas[i].insertar_registro()

    def generar_grafico(self):
        #obtenemos las ubicaciones de los objetivos
        objetivos = []
        for i in range(0, len(self.entorno)-1):
            for j in range(0, len(self.entorno)-1):
                if(self.entorno[i,j] == 1):
                    objetivos.append([i,j])
        objetivos = np.array(objetivos)
        #generamos la lista para graficar las partículas
        lista = []
        for i in range(0, self.n-1):
            lista.append(pd.DataFrame(self.particulas[i].obtener_registro()))
        
        df = pd.concat(lista)
        df.columns = ['it', 'x0', 'y0']
        df = df.sort_values(by=['it'], ascending=True)
        print(df.shape)
        print(df.to_string())
        fig = plt.figure(figsize=(8,5))
        plt.xlim(-10,0)
        plt.ylim(-6.5,0)

        def animate(i):
            p2 = fig.clear()
            df_i = df[df["it"] == i][["x0", "y0"]] 
            p1 = plt.scatter(objetivos[:,0], objetivos[:,1])
            p2 = plt.scatter(df_i["x0"], df_i["y0"])
            return p1,p2,

        ani = animation.FuncAnimation(fig, animate, repeat = True, blit = True)
        plt.show()