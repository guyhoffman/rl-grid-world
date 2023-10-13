import numpy as np
import policy
from collections import defaultdict
from base_agent import BaseAgent

# -----------------------------------------------------
# ------------ First-Visit MC Value Prediction --------
# -----------------------------------------------------
class FVMCPrediction(BaseAgent):

    def __init__(self, alpha, discount, env):
        super().__init__(alpha, discount, env)

        self.A = env.action_space
        self.explore_policy = policy.FixedRandomPolicy(env.state_space, env.action_space)
        self.draw_policy = self.explore_policy

        self.returns = defaultdict(list)
        self.episode = []

    def update(self, state, action, reward, next_state, terminal):

        self.episode.append((state, action, reward))

        if terminal:
            G = 0
            episode_states = [st for st, _, _ in self.episode]
            for idx, (s, a, r) in list(enumerate(self.episode))[::-1]:
                G = self.discount * G + r
                print (idx, ": s=", s, ", r=", r, "G=", G)
                if s not in episode_states[:idx]:
                    self.returns[s].append(G)
                    print("Appending to state", s, "return", G)
                    for a in range(self.A):
                        self.qvalues[s][a] = np.mean(self.returns[s])
            
            self.episode = []
            return True
        else:
            return False

# -----------------------------------------------------
# ------------ First-Visit MC Q-Value Prediction ------
# -----------------------------------------------------

class FVMCQPrediction(FVMCPrediction):

    # # Uncomment for exploring starts
    # def get_action(self, state):
    #     if len(self.episode) == 0:
    #         return np.random.choice(self.A)
    #     else:
    #         return BaseAgent.get_action(self, state)
    

    def update(self, state, action, reward, next_state, terminal):

        self.episode.append((state, action, reward))

        if terminal:
            G = 0
            stateactions = [(st, ac) for st, ac, _ in self.episode]
            for idx, (s, a, r) in list(enumerate(self.episode))[::-1]:
                G = self.discount * G + r
                print (f"{idx}: s={s}, a={a}, r={r}, G={G}")
                if (s,a) not in stateactions[:idx]:
                    self.returns[(s,a)].append(G)
                    print("Appending to state/action", s, a, "return", G)
                    self.qvalues[s][a] = np.mean(self.returns[(s,a)])
            
            self.episode = []
            return True
        else:
            return False

# -----------------------------------------------------
# ------------ First-Visit MC Control -----------------
# --- Set randome starting position and uncomment  ----
# --- code below for Exploring Starts -----------------
# -----------------------------------------------------

class FVMCControl(BaseAgent):

    def __init__(self, alpha, discount, env):
        super().__init__(alpha, discount, env)

        self.ssp = env.state_space
        self.asp = env.action_space
        
        self.optimal_policy = policy.GreedyPolicy(self.ssp, self.asp, self.qvalues)
        self.explore_policy = self.optimal_policy
        self.draw_policy = self.optimal_policy

        self.returns = defaultdict(list)
        self.episode = []

    # # Uncomment for exploring starts
    # def get_action(self, state):
    #     if len(self.episode) == 0:
    #         return np.random.choice(self.A)
    #     else:
    #         return BaseAgent.get_action(self, state)

    def update(self, state, action, reward, next_state, terminal):

        self.episode.append((state, action, reward))

        if terminal:
            G = 0
            stateactions = [(st, ac) for st, ac, _ in self.episode]
            for idx, (s, a, r) in list(enumerate(self.episode))[::-1]:
                G = self.discount * G + r
                print (idx, ": r=", r, "G=", G)
                if (s,a) not in stateactions[:idx]:
                    self.returns[(s,a)].append(G)
                    print("Appending to state/action", s, a, "return", G)
                    self.qvalues[s][a] = np.mean(self.returns[(s,a)])
            
            self.episode = []
            return True
        else:
            return False


# -----------------------------------------------------
# ------------ First-Visit MC Control -----------------
# --- with Epsilon-soft exploration   -----------------
# -----------------------------------------------------


class FVMCEpsiControl(BaseAgent):

    def __init__(self, alpha, discount, env, episilon=0.2):
        super().__init__(alpha, discount, env)

        ssp = env.state_space
        asp = env.action_space
        
        self.optimal_policy = policy.GreedyPolicy(ssp,asp,self.qvalues)
        self.explore_policy = policy.EpsilonGreedyPolicy(ssp, asp, self.qvalues, episilon)
        self.draw_policy = self.optimal_policy

        self.returns = defaultdict(list)
        self.episode = []

    def update(self, state, action, reward, next_state, terminal):

        self.episode.append((state, action, reward))

        if terminal:
            G = 0
            stateactions = [(st, ac) for st, ac, _ in self.episode]
            for idx, (s, a, r) in list(enumerate(self.episode))[::-1]:
                G = self.discount * G + r
                print (idx, ": r=", r, "G=", G)
                if (s,a) not in stateactions[:idx]:
                    self.returns[(s,a)].append(G)
                    print("Appending to state/action", s, a, "return", G)
                    self.qvalues[s][a] = np.mean(self.returns[(s,a)])
            
            self.episode = []
            return True
        else:
            return False

class OffPolicyMCControl(BaseAgent):

    def __init__(self, alpha, discount, env, epsilon=0.2):
        super().__init__(alpha, discount, env)

        ssp = env.state_space
        asp = env.action_space

        self.sumweights = np.zeros((ssp, asp), np.float32)

        self.optimal_policy = policy.GreedyPolicy(ssp, asp, self.qvalues)
        self.explore_policy = policy.EpsilonGreedyPolicy(ssp, asp, self.qvalues, epsilon)
        self.draw_policy = self.optimal_policy

        self.episode = []

    def update(self, state, action, reward, next_state, terminal):

        self.episode.append((state, action, reward))

        if terminal:
            G = 0
            W = 1
            stateactions = [(st, ac) for st, ac, _ in self.episode]
            for idx, (s, a, r) in list(enumerate(self.episode))[::-1]:
                G = self.discount * G + r
                print (idx, ": r=", r, "G=", G)
                self.sumweights[s][a] += W
                self.qvalues[s][a] += (W / self.sumweights[s][a]) * (G - self.qvalues[s][a])
                if self.optimal_policy.get_best_action(s) != a:
                    break

                W *= 1 / self.explore_policy.prob(s, a)
            
            self.episode = []
            return True
        else:
            return False



# -----------------------------------------------------
# ------------ Off-policy MC Control  -----------------
# -----------------------------------------------------

# -----------------------------------------------------
# ------------ ---- TD Agent -----------------------
# -----------------------------------------------------

class TDAgent(BaseAgent):

    def update(self, state, action, reward, next_state, terminal):

        # Q(s,a) = (1.0 - alpha) * Q(s,a) + alpha * (R + discount * V(s'))

        if terminal==True:
            qval_dash = 0
        else:
            qval_dash = reward + self.discount * self.get_next_value(next_state)
            
        qval_old = self.qvalues[state][action]      
        qval_new = (1.0 - self.alpha) * qval_old + self.alpha * qval_dash

        self.qvalues[state][action] = qval_new
        return terminal


class MCValueEstimator(BaseAgent):
    def __init__(self, alpha, discount, env):
        super().__init__(alpha, epsilon, discount, env)

        self.policy = policy.FixedRandomPolicy(env.state_space, env.action_space) 
        self.explore_policy = self.policy 
    
    def update(self, state, action, reward, next_state, next_action, done):
        qval_old = self.qvalues[state][action]      
        qval = reward + self.discount * ((1.0 - self.alpha)* qval_old + self.alpha * self.qvalues[next_state][action])
        self.qvalues[state][action] = qval




# -----------------------------------------------------
# ------------ ---- Q-Learning Agent -----------------------
# -----------------------------------------------------

class QLearningAgent(TDAgent):

    def __init__(self, alpha,  discount, environment, epsilon=0.2):
        super().__init__(alpha, discount, environment)

        self.optimal_policy = policy.GreedyPolicy(environment.state_space, environment.action_space, self.qvalues)
        self.explore_policy = policy.EpsilonGreedyPolicy(environment.state_space, environment.action_space, self.qvalues, epsilon)

    def get_next_value(self, next_state):
        return self.qvalues[next_state][self.get_optimal_action(next_state)]


# -----------------------------------------------------
# ------------------------- SARSA Agent -------------------------
# -----------------------------------------------------
class SARSAAgent(TDAgent):

    def __init__(self, alpha, discount, env, epsilon=0.2):
        super().__init__(alpha, discount, env)

        self.explore_policy = policy.EpsilonGreedyPolicy(
            env.state_space, 
            env.action_space, 
            self.qvalues, 
            epsilon
            )

        self.next_action = self.get_explore_action(env.get_state())

    def get_action(self, state):
        # I already know my next action since I chose it in the update
        return self.next_action

    def update(self, state, action, reward, next_state, terminal):

        # choose and remember next action
        self.next_action = self.get_explore_action(next_state)
        
        return super().update(state, action, reward, next_state, terminal)

    def get_next_value(self, next_state):
        return self.qvalues[next_state][self.next_action]


# -----------------------------------------------------
# ------------------- n-Step SARSA Agent ------------------
# -----------------------------------------------------
class NStepSARSAAgent(BaseAgent):

    def __init__(self, alpha, discount, environment, n, epsilon=0.2):
        super().__init__(alpha, discount, environment)

        self.n = n
        self.epsilon=epsilon
        self.policy = policy.EpsilonGreedyPolicy(
            environment.state_space, environment.action_space, 
            self.qvalues, self.epsilon)
        self.explore_policy = self.policy

        self.init_episode(environment)

    def init_episode(self, env):
        self.states = np.full(self.n+1, np.nan)
        self.actions = np.full(self.n+1, np.nan)
        self.rewards = np.full(self.n+1, np.nan)
        
        state = env.get_state()
        self.states[0] = state

        action = self.get_explore_action(state)

        self.actions[0] = action
        self.next_action = action

        self.T = np.infty
        self.t = 0
        self.tau = 0

    def get_action(self, state):
        return self.next_action

    def update(self, state, action, reward, next_state, terminal):
        if (self.tau < self.T - 1):

            self.rewards[(self.t + 1) % self.n] = reward
            self.states[(self.t + 1) % self.n] = next_state

            # We're in a terminal state, need to unroll the rewards
            if terminal == True:
                self.T = self.t + 1
            else:
                # Sample next state to use
                self.next_action = self.get_explore_action(next_state)
                self.actions[(self.t + 1) % self.n] = self.next_action
        
            tau, T, t, n = self.tau, self.T, self.t, self.n
            discount, alpha = self.discount, self.alpha
            states, actions, qvalues = self.states, self.actions, self.qvalues

            tau = t - n + 1
            if tau >= 0:  # We have at least n observations
                G = 0
                # Add actual rewards to return
                for i in range(tau+1, min(tau+n, T)):
                    G += discount**(i-tau-1) * self.rewards[i % n]
                
                # Add Q estimate to return
                if tau + n < T:
                    
                    idx = (tau + n) % n
                    s, a = int(states[idx]), int(actions[idx])
                    
                    G += discount**n * qvalues[s][a] 
                    
                    idx = tau % n
                    s, a = int(states[idx]), int(actions[idx])
                    
                    qvalues[s][a] = (1 - alpha) * qvalues[s][a] + alpha * G  
            t = t + 1
            return False
        else:
            return True


# -----------------------------------------------------
# ------------------ Expected Value SARSA Agent ---------------------
# -----------------------------------------------------
    
class EVSarsaAgent(BaseAgent):
    
    def get_next_value(self, next_state):
        
        # estimate V(s) as expected value of Q(state,action) over possible actions assuming epsilon-greedy policy
        # V(s) = sum [ p(a|s) * Q(s,a) ]
          
        best_action = 0
        max_val = self.qvalues[next_state][0]
        
        for action in range(self.action_space):
            
            q_val = self.qvalues[next_state][action]
            if q_val > max_val:
                max_val = q_val
                best_action = action
        
        state_value = 0.0
        n_actions = self.action_space
        
        for action in range(self.action_space):
            
            if action == best_action:
                trans_prob = 1.0 - self.epsilon + self.epsilon/n_actions
            else:
                trans_prob = self.epsilon/n_actions
                   
            state_value = state_value + trans_prob * self.qvalues[next_state][action]

        return state_value
