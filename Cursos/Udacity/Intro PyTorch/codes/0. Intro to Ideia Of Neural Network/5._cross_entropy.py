import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y) # converter em float. y: Se o ocorreu ou nâo o evento
    P = np.float_(P) # convete em float. P =  probabilidades do evento ocorrer
    # Cross-Entropy: Negativo da soma de: acontecer o evento que ocorreu
    # Da forma que esta construida: o evento que nâo ocorrer será zero nessa soma
    # Assim, somente um dos termos (Y * np.log(P)) ou (1 - Y) * np.log(1 - P)
    # vai existir, sendo assim, nao é necessario fazer um if para saber qual evento ocorreu
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
