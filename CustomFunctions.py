import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import emcee as mcmc
import corner as mcmc_vis


#DATA CREAZIONE: 08/06/2026
#AUTORE: SIMONE RAIMONDI

#SIMONE:
#INFORMAZIONI GENERALI:
#In questo file sono definite tutte le funzioni da utilizzare per la simulazione del fenomeno del
#smFRET. Per i risultati della simulazione si veda il file ProvaSimulazioneDati.ipynb.


#-------------------------------------------------------------------------------------------------
#-------------------------------Funzioni per Simulazione Dati:------------------------------------
#-------------------------------------------------------------------------------------------------

#1) Funzione di Traiettorie:
#Definisco la funzione che generi la traiettoria degli stati partendo dalla matrice T di transizione,
#dal numero dei passi da simulare e dallo stato iniziale.
def trajectory_generation(T, n_steps=100, initial_state=0):
    #Inizializzazione degli stati.
    states = [initial_state]

    for _ in range(n_steps): #ciclo for per generare la traiettoria a seconda del numero di step
        s = states[-1]
        s_next = np.random.choice(T.shape[0], p=T[s])  # Random walk sulle transizioni
        states.append(s_next) #Percorso della traiettoria.
    return np.array(states) #Output convertito in array numpy per comodità di utilizzo.

#2) Funzione di simulazione dei fotoni:
#Definisco la funzione che calcola la simulazione dei fotoni partendo dalla traiettora,
#dalle probabilità di FRET dal numero di fotoni per bin e distinguendo il caso con rumore e senza.
def photon_simulation(trajectory, fret_probabilities, photons_per_bin=100, noise=False):
    n_G = np.zeros(len(trajectory))
    n_R = np.zeros(len(trajectory))

    for i, state in enumerate(trajectory):
        mu_G = photons_per_bin * (1 - fret_probabilities[state])  # Media fotoni verdi
        mu_R = photons_per_bin * fret_probabilities[state]        # Media fotoni rossi

        if noise:
            n_G[i] = np.random.poisson(mu_G)  # Simulazione con rumore
            n_R[i] = np.random.poisson(mu_R)
        else:
            n_G[i] = mu_G  # Simulazione senza rumore (valori medi)
            n_R[i] = mu_R

    FRET_Obs_R = n_R / (n_G + n_R + 1e-10)  # Aggiungo un piccolo termine per evitare divisione per zero
    # FRET_Obs_G = n_G / (n_G + n_R + 1e-10) #Definizione formale della quantità, ma anche in questo caso
                                             # sono variabili normalizzate, perciò FRET_Obs_G = 1 - FRET_Obs_R
    FRET_Obs = [FRET_Obs_R, 1-FRET_Obs_R]  # FRET osservato per rosso e verde
    return n_G, n_R, np.array(FRET_Obs)


#3) Funzione di calcolo del Transition Density Plot (TDP):
#Definisco la funzione che calcoli TDP a partire dalla traiettoria e dai FRET osservati.
def TDP(trajectory, FRET_Obs):
    FRET_before = []
    FRET_after = []
    for i in range(len(trajectory) - 1):
        if trajectory[i] != trajectory[i + 1]:  # Cambio di stato
            FRET_before.append(FRET_Obs[0][i])  # FRET prima del cambio di stato
            FRET_after.append(FRET_Obs[0][i + 1])  # FRET dopo il cambio di stato
    return np.array(FRET_before), np.array(FRET_after)

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------