import pandas as pd
import random

class EpsilonGreedy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def decide(self):
        r = random.random()
        if(r < self.epsilon):
            return True
        return False
    
class Greedy(EpsilonGreedy):
    def __init__(self):
        super().__init__(1)
    
    def decide(self):
        return False
    
class ExploreOnly(EpsilonGreedy):
    def __init__(self):
        super().__init__(0)
    
    def decide(self):
        return True

class QTable:
    def __init__(self, alpha, gamma, acoes, estado_inicial, policy, retornarPrimeiroAoExploitar=True):
        self.alpha = alpha
        self.gamma = gamma
        self.estado_atual = estado_inicial
        self.acoes = acoes
        self.policyClass = policy
        self.retornarPrimeiroAoExploitar = retornarPrimeiroAoExploitar
        self.recompensa_total= 0

    def __str__(self):
        return self.QTable.T.to_string()

    #Inicializa a QTable
    def inicializaQTable(self, QTable):
        if isinstance(QTable, pd.DataFrame):
            self.QTable = QTable
        else:
            self.QTable = pd.DataFrame(QTable)

    #carrega a QTable a partir de um csv
    def carregarQTable(self, arquivo):
        tabela = pd.read_csv(arquivo)
        tabela = tabela.T
        tabela = tabela.rename(columns=tabela.iloc[0])
        tabela = tabela[1:]
        tabela = tabela.astype(float)

        self.inicializaQTable(tabela)

    #Atualiza o peso e aprende
    def atualizaPeso(self, a, prox_estado, recompensa):
        self.QTable = self.QTable.copy()
        self.recompensa_total += recompensa
        pd.options.mode.chained_assignment = None
        self.QTable[self.estado_atual][a] = self.QTable[self.estado_atual][a] + self.alpha * (recompensa + self.gamma * self.QTable[prox_estado].max() - self.QTable[self.estado_atual][a])
        pd.options.mode.chained_assignment = 'warn'
        self.estado_atual = prox_estado

    #Escolhe uma ação aleatoria
    def explore(self):
        return random.choice(self.acoes)
    
    #Escolhe a(s) melhor(es) ação(ões) disponíveis no momento
    def exploit(self):
        s = self.QTable[self.estado_atual]

        # Encontre a melhor ação possível
        if(self.retornarPrimeiroAoExploitar):
            return s.argmax()
        
        # Busca as melhores ações
        max_valor = s.max()

        indices_max_valor = s[s == max_valor].index

        return indices_max_valor
    
    def policy(self):
        if(self.policyClass.decide()):
            return self.explore()
        else:
            # print('exploita')
            return self.exploit()
        
    def salvar(self, nome_arquivo):
        self.QTable.T.to_csv(nome_arquivo)