# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 18:18:06 2023

@author: Working-Ntb

Version más eficiente para rondas mayores a 4
"""

from jugador import Player
from modelo_IA import Player_IA
# import random
import pandas as pd
import numpy as np
# import tensorflow as tf
from tensorflow.keras import backend
import time
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# %%


def play_ultimatum(player1, player2, amount, dynamic=False):
    # if player1.min_wta is None:
    #     player1.min_wta = calculate_min_wta(player1.history)
    # if player2.min_wta is None:
    #     player2.min_wta = calculate_min_wta(player2.history)
    # amount=100

    offer = player1.make_offer(amount)
    acceptance = player2.accept(offer, amount)
# =============================================================================
# Se utiliza una lista de listas temporal que al finalizar los ciclos
# se alamacena. Esto permite ganar eficienia ya que el método .loc de pandas
# es muy lento
# Columnas de history: ['Offerer', 'Offer', 'Acceptance', 'min_wta', 'max_wtp']
#                       proximas: ['earning', 'cumul_ear']
# =============================================================================
    player1.temp_history.append([1, offer, int(acceptance),
                                 player1.min_wta, player1.max_wtp])
    player2.temp_history.append([0, offer, int(acceptance),
                                 player2.min_wta, player2.max_wtp])

    if acceptance:
        player1.earnings += amount - offer
        player2.earnings += offer

    if dynamic:
    # =========================================================================
    # Intentaba que los jugadores suban o bajen 1% en funcion de si había trato
    # o no. El exito llevaba a buscar ganar más y el rechazo, menos. Pero el
    # resultado es convergencia a (0,5 ; 0,5)
    # =========================================================================
        if acceptance:
            player1.max_wtp = max(player1.max_wtp - 0.01, 0)
            player2.min_wta = min(player2.min_wta + 0.01, 1)
        else:
            player1.max_wtp = min(player1.max_wtp + 0.01, 1)
            player2.min_wta = max(player2.min_wta - 0.01, 0)

    # player1.utility(offer, player2.earnings)
    # player2.utility(player2.earnings, offer)


# player1 = Player(name=1, min_wta=random.uniform(0, 1), max_wtp=random.uniform(0, 1))
# player2 = Player(name=2, min_wta=random.uniform(0, 1), max_wtp=random.uniform(0, 1))

# play_ultimatum(player1, player2, 100)
# %%

def play_random_round(list_players, amount):

    np.random.shuffle(list_players)

    for i in range(0, len(list_players), 2):
        player1 = list_players[i]
        player2 = list_players[i+1]
        # print(f'{player1.name} vs {player2.name}')
        play_ultimatum(player1, player2, amount)

def play_balanced_round(group1, group2, amount):

    np.random.shuffle(group1)
    np.random.shuffle(group2)

    # print(len(group1))
    # print(len(group2))
    # for p in group1:
    #     print(p.name)
    # print('')
    # for p in group2:
    #     print(p.name)

    for i in range(len(group1)):
        player1 = group1[i]
        player2 = group2[i]
        # print(f'{player1.name} vs {player2.name}')
        play_ultimatum(player1, player2, amount)


def play_cycle(list_players, list_ia, amount, balanced=True):

    if balanced:
        # print( list_players[0])
        # print( list_players[1])
        # print( list_ia[0])
        # print( list_ia[1])

        grupo1 = list_players[0] + list_ia[0]
        grupo2 = list_players[1] + list_ia[1]
        for i in range(n_rounds//2):
            print(f'Round: {i+1}')
            play_balanced_round(grupo1, grupo2, amount)
            # print('Player {player1.name}: \n player1.history.loc{n_rounds-1}')

        for i in range(n_rounds//2, n_rounds):
            print(f'Round: {i+1}')
            play_balanced_round(grupo2, grupo1, amount)
    # print('Player {player1.name}: \n player1.history.loc{n_rounds-1}')

        for ia in list_ia[0]:
            ia.min_wtp()
        for ia in list_ia[1]:
            ia.min_wtp()

    else:
        for i in range(n_rounds):
            print(f'Round: {i+1}')
            play_random_round(list_players[0], amount)

        for ia in list_ia:
            ia.min_wtp()

    # if balanced:
    #     total_players = list_players[0] + list_ia[0] +\
    #         list_players[1] + list_ia[1]
    # else:
    #     total_players = list_players + list_ia

    # for player in total_players:
    #     new_info = pd.DataFrame(player.temp_history)
    #     # print(new_info)
    #     player.history = pd.concat([player.history, new_info])


def experiment(n_players, n_rounds, amount, n_cycle=1,
               min_wta=None, max_wtp=None, balanced=True,
               n_ia=0, clone_ia=False, seed=None):

    if seed is None:
        seed = np.random.seed(np.randint)
    print(f'La semilla es: {seed}')

    inicio = time.time()

    if (n_ia * (1 + clone_ia) + n_players) % 2 == 1:
        n_players += 1

    if (min_wta is None) & (max_wtp is None):
        players = [Player(name=i+1, egoism=np.random.uniform(0.3, 1),
                          min_wta=np.random.normal(0.35, 0.05),
                          max_wtp=np.random.normal(0.35, 0.05))
                   for i in range(n_players)]

    elif min_wta is None:
        players = [Player(name=i+1, egoism=np.random.uniform(0.01, 0.99),
                          min_wta=np.random.normal(0.35, 0.05),
                          max_wtp=np.max_wtp)
                   for i in range(n_players)]

    elif max_wtp is None:
        players = [Player(name=i+1, egoism=np.random.uniform(0.3, 1),
                          min_wta=min_wta,
                          max_wtp=np.random.normal(0.35, 0.05))
                   for i in range(n_players)]

    else:
        players = [Player(name=i+1, egoism=np.random.uniform(0.35, 0.05),
                          min_wta=min_wta,
                          max_wtp=max_wtp)
                   for i in range(n_players)]

    if n_ia:
        ia_players = [Player_IA(name='ia'+str(+1+i),
                                min_wta=np.random.normal(0.35, 0.05),
                                max_wtp=0.2,
                                layer_of=2,  # np.random.randint(1, 4)
                                neurons_of=16)
                      for i in range(n_ia)]
        if clone_ia:
            clones_ia = [Player(name='ia'+str(+1+i)+'_clone',
                                min_wta=ia_players[i].min_wta,
                                max_wtp=ia_players[i].max_wtp)
                         for i in range(n_ia)]
            players += clones_ia

    # for i in ia_players:
    #     print(i.name)

    list_players = []
    list_ia = []

    if balanced:
        n = len(players) // 2

        group1 = players[:n]
        list_players.append(group1)

        group2 = players[n:]
        list_players.append(group2)

    else:
        list_players.append(players)

    if n_ia:
        if balanced:
            m = len(ia_players) // 2
            ia1 = ia_players[m:]
            list_ia.append(ia1)
            ia2 = ia_players[:m]
            list_ia.append(ia2)
        else:
            list_ia.append(ia_players)
    else:
        list_ia = [[], []]

    # print(list_ia, end='\n')
    # print(list_players)

    if n_rounds % 2 == 1:
        n_rounds += 1

    test_time = []

    for i in range(n_cycle):
        print(f'cycle: {i+1}')
        ini_cycle = time.time()
        play_cycle(list_players, list_ia, amount, balanced=balanced)
        time_cycle = time.time()-ini_cycle
        if i % 5 == 0:
            backend.clear_session()
        print(f'El ciclo tardó {time_cycle:0.6f}s')
        test_time.append(time_cycle)
        # if time.time() - inicio > 1800:
        #     return players + ia_players
        print()

    total_players = players + ia_players
    for player in total_players:
        player.history =\
            pd.DataFrame(player.temp_history,
                         columns=['Offerer', 'Offer', 'Acceptance', 'min_wta',
                                  'max_wtp'])
        player.history['earning'] =\
            np.where(player.history['Acceptance'].values == 1,
                     player.history['Offer'].values, 0)
        player.history['cumul_earn'] = player.history['earning'].values.cumsum()

    fin = time.time()
    print(fin - inicio)  # 1.0005340576171875

    return total_players, test_time


#%%

def df_result(result):
    names = [player.name for player in result]
    df = pd.DataFrame(names, columns=['names'])
    df['names'] = df['names'].astype(str)
    df['earnings'] = [player.earnings for player in result]
    df['min_wta'] = [player.min_wta for player in result]
    df['max_wtp'] = [player.max_wtp for player in result]
    df['offerer_rate'] =\
        [player.history['Offerer'].mean() for player in result]
    df['mean_off_recibed'] =\
        [player.history[player.history['Offerer']==0]['Offer'].mean() for player in result]
    df['mean_off_sent'] =\
        [player.history[player.history['Offerer']==1]['Offer'].mean() for player in result]
    return df

#%%

# random.seed(91218)


# %% IA experiments

n_total = 1000
n_ia = 0
clone_ia = True
n_players = n_total - n_ia * (1 + clone_ia)
n_rounds = 20
amount = 100
n_cycle = 300


#%%
result = experiment(n_players, n_rounds, amount, n_cycle,
                    n_ia=n_ia, clone_ia=clone_ia,
                    seed=181222)
df = df_result(result[0])
df = df.sort_values('earnings', ascending=False)

dic_ia = {result[0][-i-1].name:result[0][-i-1].history for i in range(n_ia)}

#%%
# result2 = experiment(n_players, n_rounds, amount, n_cycle,
#                     n_ia=n_ia, clone_ia=clone_ia,
#                     seed=181222)
# df2 = df_result(result2[0])
# df2 = df2.sort_values('earnings', ascending=False)

# dic_ia2 = {i:result2[0][-i-1].history for i in range(n_ia)}


#%%

# two_types = [np.random.normal(0.3, 0.1),
#              np.random.lognormal(0.01,0.5)][np.random.randint(0,2)]

# # result2 = experiment(n_players, n_rounds, amount,
#                       # n_cycle=n_cycle, min_wta=0.01)
# result3 = experiment(n_players, n_rounds, amount, n_cycle,
#                      min_wta=two_types,
#                      max_wtp=two_types,
#                     n_ia=n_ia, clone_ia=clone_ia)
# df3 = df_result(result3[0])
# df3 = df3.sort_values('earnings', ascending=False)

# dic_ia3 = {i:result3[0][-i-1].history for i in range(n_ia)}

#%%


plt.figure(1)
g1 = sns.scatterplot(
    data=df, x="max_wtp", y="earnings", size="min_wta",
    sizes=(20, 200), hue="min_wta", palette='Greens_r')  #, hue_norm=(0, 7) , legend="full"


# g2 = sns.scatterplot(
#     data=df2, x="max_wtp", y="earnings", size="min_wta",
#     sizes=(20, 200), palette='Blues_r', hue="min_wta")  # , hue_norm=(0, 7) , legend="full"

# g3 = sns.scatterplot(
#     data=df3, x="max_wtp", y="earnings", size="min_wta",
#     sizes=(20, 200), palette='Reds_r', hue="min_wta")  # , hue_norm=(0, 7) , legend="full"

plt.show()

# %% Analizar tiempo

plt.figure(2)
times1 =plt.plot(np.array(result[1]))
# times2 =plt.plot(np.array(result2[1]))

plt.show()

# %% Mostrar permormance deia y sus clones

# df_ia = pd.DataFrame()

# for i 

#%% Mostrar la evolución de la wta y wtp  de los 12 mejores


# top_12 = df.index.to_list()[:12]
# df_top = pd.DataFrame({'time': pd.Series(dtype='int'),
#                         'player': pd.Series(dtype='str'),
#                         'earnings': pd.Series(dtype='float'),
#                         'min_wta': pd.Series(dtype='float'),
#                         'max_wtp': pd.Series(dtype='float')})
# time_list = [i for i in range(n_rounds*n_cycle)]

# for i in top_12:
#     # i=412
#     earnings = result[i].history[['Offer', 'Acceptance']].copy()
#     earnings["Acceptance"] = earnings["Acceptance"].astype(int)
#     earnings['earnings'] = earnings["Offer"] * earnings["Acceptance"]
#     earnings = earnings['earnings'].cumsum()
#     df_temp = pd.DataFrame({'time': time_list,
#                             'player': i,
#                             'earnings': earnings,
#                             'min_wta': result[i].history.min_wta,
#                             'max_wtp': result[i].history.max_wtp})
#     df_top = pd.concat([df_top, df_temp], ignore_index=True)


# sns.set_theme(style="dark")
# # Plot each year's time series in its own facet
# g = sns.relplot(
#     data=df_top,
#     x="time", y="earnings", col="player", hue="player",
#     kind="line", palette="crest", linewidth=4, zorder=5,
#     col_wrap=3, height=2, aspect=1.5, legend=False,
# )

# # Iterate over each subplot to customize further
# for player, ax in g.axes_dict.items():

#     # Add the title as an annotation within the plot
#     ax.text(.8, .85, player, transform=ax.transAxes, fontweight="bold")

#     # Plot every year's time series in the background
#     sns.lineplot(
#         data=df_top, x="time", y="earnings", units="player",
#         estimator=None, color=".7", linewidth=1, ax=ax,
#     )

# # Reduce the frequency of the x axis ticks
# ax.set_xticks(ax.get_xticks()[::2])

# # Tweak the supporting aspects of the plot
# g.set_titles("")
# g.set_axis_labels("", "Players")
# g.tight_layout()
# plt.show()

# %% To free my resources, I use:

# import os, signal
# os.kill(os.getpid(), signal.SIGTERM )
