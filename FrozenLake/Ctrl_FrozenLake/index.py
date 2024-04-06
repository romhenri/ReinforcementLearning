import gymnasium as gym
from QTable import QTable, EpsilonGreedy
import numpy as np

print_age = True
lr = 0.001
gamma = 0.4
actions = [0,1,2,3]
e0 = 0
policy = EpsilonGreedy(0.4)

# Exploration
def train_agent(
    epochs = 5000
):
    tabela = np.zeros((4, 16))
    Q = QTable(lr, gamma, actions, e0, policy)
    Q.inicializaQTable(tabela)

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    rewards = []
    for i in range(epochs):
        if print_age:
            print(f'Época atual: {i+1}')

        observation, info = env.reset(seed=123, options={})
        Q.estado_atual = observation

        done = False

        # Play the game until it's done
        while not done:
            action = env.action_space.sample()  # agent policy that uses the observation and info
            
            # Observation is the current state
            observation, reward, terminated, truncated, info = env.step(action)
            Q.atualizaPeso(action, observation, reward) # Atualiza o peso e aprende 

            done = terminated or truncated
        
            rewards.append(reward)

    Q.salvar('QTable.csv')
    print(Q)
    env.close()

# Exploitation
def play_game(
    times = 1,
    qtabble = 'QTable.csv'
):
    tabela = np.zeros((4, 16))
    Q = QTable(lr, gamma, actions, e0, policy)
    Q.inicializaQTable(tabela)
    Q.carregarQTable(qtabble)

    for time in range(times):
        env_play = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')

        observation, info = env_play.reset(seed=123, options={})
        Q.estado_atual = observation
        # Use the csv QTable

        done = False

        # Play
        while not done:
            action = Q.exploit()  # agent policy that uses the observation and info
            
            #observation é meu estado atual
            observation, reward, terminated, truncated, info = env_play.step(action)
            Q.estado_atual = observation

            done = terminated or truncated

train_agent(5000)
play_game(3, 'QTable.csv')