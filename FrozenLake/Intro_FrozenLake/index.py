import gymnasium as gym
from QTable import QTable, EpsilonGreedy
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

print_age = True

# Learning Rate é a taxa de aprendizado
lr = 0.001
# Gamma é o fator de desconto
gamma = 0.4
# Actions são as ações que o agente pode tomar
actions = [0,1,2,3]
# e0 é a probabilidade de explorar
e0 = 0
# Epsilon Greedy é a política de exploração do agente
policy = EpsilonGreedy(0.4)
# Épocas são os episódios que o agente vai jogar
epochs = 5000

tabela = np.zeros((4, 16))

Q = QTable(lr, gamma, actions, e0, policy)
Q.inicializaQTable(tabela)

print(Q.QTable)

# Treinamento
rewards = []
for i in range(epochs):
    if print_age:
        print(f'Época atual: {i+1}')

    observation, info = env.reset(seed=123, options={})
    Q.estado_atual = observation

    done = False

    while not done:
        action = env.action_space.sample() # O agente escolhe uma ação aleatória entre as possíveis
        
        # observation é meu estado atual
        observation, reward, terminated, truncated, info = env.step(action)
        Q.atualizaPeso(action, observation, reward) # Atualiza o peso e aprende 

        done = terminated or truncated
        
    rewards.append(reward)

env.close()

print(Q.QTable)

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')

observation, info = env.reset(seed=123, options={})
Q.estado_atual = observation

done = False

# Play
while not done:
    action = Q.exploit()  # O agente escolhe a melhor ação
    
    # observation é meu estado atual
    # reward é a recompensa que eu recebo por ter feito a ação
    observation, reward, terminated, truncated, info = env.step(action)
    Q.estado_atual = observation

    done = terminated or truncated

Q.salvar('QTable.csv')
env.close()