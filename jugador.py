# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:10:26 2023

@author: Working-Ntb
"""

import pandas as pd
import numpy as np

class Player:
    def __init__(self, name, min_wta=0, max_wtp=0.5, egoism=1):
        self.name = name
        self.earnings = 0
        self.temp_history = []
        self.history = pd.DataFrame(columns=['Offerer', 'Offer', 'Acceptance',
                                             'min_wta', 'max_wtp', 'earning',
                                             'cumul_ear'])
        self.egoism = round(egoism, 4)
        self.min_wta = max(0.01, round(min_wta, 2))
        self.max_wtp = min(round(max_wtp, 2), 0.5)
        self.last_utility = 0

    def utility(self, individual_gain, collective_gain, utility_type='linear'):
        if utility_type == 'linear':
            return self.egoism * individual_gain +\
                (1 - self.egoism) * collective_gain
        elif utility_type == 'cobb-douglas':
            return (individual_gain ** self.egoism) *\
                (collective_gain ** (1 - self.egoism))
        else:
            raise ValueError("Invalid utility type: '{}'".format(utility_type))

    def make_offer(self, amount=1, function=None):
        shared_part = self. max_wtp
        if function == 'Uniform':
            shared_part = np.random.uniform(0, self.max_wtp)
        offer = amount * shared_part
        return offer

    def accept(self, offer, amount=1):
        '''Decide whether to accept the offer based on the player's history
        of offers and acceptances'''

        min_wta = self.min_wta * amount
        acceptation = False

        if offer >= min_wta:
            acceptation = True

        return acceptation
