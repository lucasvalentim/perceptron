import numpy as np


class Perceptron(object):
    def __init__(self, taxa_de_apredizagem=0.05, max_epocas=100, vies=1, peso_do_vies=0, x_pesos=None):
        self.__taxa_de_apredizagem = taxa_de_apredizagem
        self.__max_epocas = max_epocas
        self.__vies = vies
        self.__peso_do_vies = peso_do_vies
        self.__x_pesos = x_pesos
    
    @property
    def peso_do_vies(self):
        return self.__peso_do_vies
        
    @property
    def x_pesos(self):
        return self.__x_pesos
    
    @property
    def percentual_de_acertos(self):
        return self.__percentual_de_acertos
        
    def treinar(self, X_treino, y_treino):
        self.__x_pesos = np.zeros(X_treino.shape[1])
        
        for i in range(len(X_treino)):
            n_epocas = 0
            
            while n_epocas < self.__max_epocas: 
                u = self.__peso_do_vies * self.__vies

                for j in range(len(self.__x_pesos)):
                    u += self.__x_pesos[j] * X_treino[i][j]

                y = 1 if u >= 0 else 0
                e = y_treino[i] - y
                
                if e == 0:
                    break

                self.__peso_do_vies += self.__taxa_de_apredizagem * e * self.__vies
                
                for j in range(len(self.__x_pesos)):
                    self.__x_pesos[j] += self.__taxa_de_apredizagem * e * X_treino[i][j]
                    
                n_epocas += 1
                    
    def teste(self, X_teste, y_teste):
        n_erros = 0

        for i in range(len(X_teste)):
            u = self.__peso_do_vies * self.__vies

            for j in range(len(self.__x_pesos)):
                u += self.__x_pesos[j] * X_teste[i][j]

            y = 1 if u >= 0 else 0
            e = y_teste[i] - y

            if e != 0:
                n_erros += 1

        self.__percentual_de_acertos = 1 - n_erros / len(X_teste)
