from MAB import *
import matplotlib.pyplot as plt


class FiniteSets(GenericMAB):
    def __init__(self, method, param, k, alpha, beta, gamma):
        """
        Initialization of Finite Set Bandit Problems : theta in [1,L], Y in [1,N], A in [1,K]
        K is the number of arms in our algorithm and is denoted nb_arms
        :param method: list, distributions of each arm
        :param param: list, parameters of each arm's distribution
        :param k: monte-carlo, k Q tables.
        :param alpha: learning rate of Q table
        :param beta: computing uncertainty for regret
        :param gamma: discount of reward for Q table
        """
        super().__init__(method, param)
        self.flag = False
        self.optimal_arm = None

        self.Q = np.random.random([k, self.nb_arms])
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def IDS_action(self):
        pass

    def choose_arm(self):
        mean = np.mean(self.Q, axis=-2)
        zero_mean = self.Q - np.expand_dims(mean, axis=-2)
        var = np.mean(np.square(zero_mean), axis=-2)
        std = np.sqrt(var)
        regret = np.max(mean + self.beta * std, axis=-1, keepdims=True)
        regret = regret - (mean - self.beta * std)
        regret_sq = np.square(regret)
        info_gain = np.log(1 + var / np.mean(self.vars)) + 1e-5
        ids_score = regret_sq / info_gain
        action = np.argmin(ids_score, axis=-1)
        return action

    def update_Q(self, arm, reward):
        self.Q[arm, :] += self.alpha * (reward + self.gamma * np.max(self.Q, axis=-2) - self.Q[arm, :])

    def IDS_Qlearn(self, T):
        Sa, Na, Y, arm_sequence = self.init_lists(T)
        reward = np.zeros(T)

        for t in range(T):
            if not self.flag:
                arm = self.choose_arm()
            else:
                arm = self.optimal_arm
            self.update_lists(t, arm, Sa, Na, Y, arm_sequence)
            reward[t] = Y[t]
            self.update_Q(arm, reward[t])
        return reward, arm_sequence

    def mc_regret(self, N, T):
        mc_regret = 0
        for _ in tqdm(range(N), desc='Computing ' + str(N) + ' simulations'):
            reward, arm_sequence = self.IDS_Qlearn(T)
            print(np.mean(reward))
            mc_regret += self.regret(reward, T)
        return mc_regret / N


if __name__ == "__main__":
    nb_arms = 5
    p = [[-1, 0.1], [0, 1], [1, 1], [2, 2], [5, 0.5]]
    my_MAB = FiniteSets(['G'] * nb_arms, p, k=10, alpha=0.001, beta=0.1, gamma=0.9)
    regret_IDS = my_MAB.mc_regret(N=10, T=10000)
    plt.plot(regret_IDS, label='IDS', c='cyan')
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Time horizon')
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

