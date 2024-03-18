# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:57:02 2023

@author: Working-Ntb
"""

import numpy as np
from jugador import Player
from tensorflow.keras import datasets, Sequential, models, backend
from tensorflow.keras.layers import Flatten, Dense

# %%

class Player_IA(Player):
    def __init__(self, name, min_wta=0.01, max_wtp=0.5, max_train=10000,
                 layer_of=2, neurons_of=16, layer_ac=1, neurons_ac=16):
        super().__init__(name, min_wta, max_wtp)
        # self.name 
        # self.earnings
        # self.history
        # self.min_wta
        # self.max_wtp
        # Atributos propios
        # self.objective = objective
        self.layer_of = layer_of
        self.neurons_of = neurons_of
        self.offer_model =\
            self.gen_offer_model(layers=layer_of, neurons=neurons_of)
        self.acceptance_model = self.gen_acc_model()
        self.try_random = True
        self.max_train = max_train
        self.has_trained = False
        self.training = []

    def gen_offer_model(self, layers, neurons):
        print(f'Generando modelo de oferta con {layers} capas con {neurons} neuronas')
        model = Sequential()
        model.add(Dense(neurons, input_dim=2, activation='relu'))
        for i in range(layers-1, layers):
            model.add(Dense(neurons, input_dim=neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        return model

    def gen_acc_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=2, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        return model

    def min_wtp(self, amount=100, see_itself=True, max_offer=0.5):
        n = len(self.temp_history)
        if n:
            temp_history = np.array(self.temp_history)
            history = temp_history[:, 0:3]
            history = np.asarray(history).astype(np.float32)

            if not see_itself:
                history =  history[history[:, 0] == 1]  # Solo cuando es oferente

            x = history[:, 1:]
            y = history[:, 0]

            # print(x)
            # print(y)

            # if self.has_trained:
            #     self.offer_model = models.load_model(self.name)

            if len(history) < self.max_train:
                self.model_history =\
                    self.offer_model.fit(x, y,
                                         epochs=1,
                                         verbose=0)
                self.training.append(self.model_history.history)

            # prediction = 0

# =============================================================================
# Algoritmo similar a busqueda binaria para bajar los intentos
# =============================================================================

            if self.offer_model.predict(
                    np.array([[amount*max_offer, 1]])) >= 0.5:
                values = np.arange(0.01, max_offer, 0.01)
                values = values * amount

                while len(values) > 1:
                    cut = len(values)//2
                    wtp = values[cut]
                    prediction =\
                        self.offer_model.predict(np.array([[wtp, 1]]),
                                                 verbose=0)
                    result = round(prediction[0][0], 2)
                    # print(values)
                    # print(cut, values[cut])
                    # print(prediction)
                    if result < 0.5:
                        values = values[cut:]
                        # print(values)
                    elif result > 0.5:
                        values = values[:cut]
                        # print(values)
                    else:
                        values = [values[cut]]
                        # print(values)

                wtp = values[0]
                self.max_wtp = min(wtp/amount, max_offer)
                # self.offer_model.save(self.name)
                self.has_trained = True
            else:
                print(f'La nueva wtp es mayor a: {max_offer:0.2f}')
                # Si cree que debe ofrecer mas de lo maximo razonable puede:
                # Dar ese maximo
                self.max_wtp = max_offer

                # Lo minimo que acepta
                # self.max_wtp = self.min_wta

                #Buscar la ultima oferta que acepto o lo minimo que acepta
                # try:
                #     aceptadas =\
                #         temp_history[
                #             (temp_history[:, 0] == 0) & (temp_history[:, 2] == 1)
                #             ]
                #     self.max_wtp = aceptadas[-1,1] / amount
                # except IndexError:
                #     print('No aceptó ofertas, juega la min wta')
                #     self.max_wtp = self.min_wta

                #Aumentar un poco la oferta
                # self.max_wtp =\
                #     self.max_wtp + round(np.random.random() * 0.1)

                self.max_wtp = min(max_offer, self.max_wtp)

            print(f'La nueva wtp es: {self.max_wtp:0.2f}')

    def make_offer(self, amount=1, min_random=0):
        history = self.temp_history
        # print(len(history))
        # if len(history[history['Role'] == 'Offerer']) == 0:
        #     wtp = self.max_wtp
        if len(history) < min_random:
            wtp = round(np.random.uniform(0.01, 0.5), 2)
        else:
            wtp = self.max_wtp
        # print(wtp)
        return wtp * amount

    def accept(self, offer, amount=1):
        '''Decide whether to accept the offer based on the player's history
        of offers and acceptances'''

        min_wta = self.min_wta * amount
        acceptation = False

        if offer >= min_wta:
            acceptation = True

        return acceptation

'''
Evaluando posibles soluciones.
1) Que aprendan a jugar de la nada sin wtp/wta:
    -Salidas:
        Si oferente, numero entre monto y monto /100
        Si aceptante, booleano
    -Entradas:
        ganancias, ranking relativo
¡Como implemento si no tengo un ideal? min(rondas * monto * %_esperado)

Si %_esperado = 100% entonces el jugador siempre se mueve
'''

