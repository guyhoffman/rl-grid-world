import numpy as np
import random
import policy
import cv2



# -----------------------------------------------------
# -------------------------- Base Agent ----------------
# -----------------------------------------------------

class BaseAgent:
    
    def __init__(self, alpha, discount, environment):
 
        self.action_space = environment.action_space
        self.alpha = alpha
        self.discount = discount
        self.qvalues = np.zeros((environment.state_space, environment.action_space), np.float32)
        self.policy = policy.RandomPolicy(environment.state_space, environment.action_space)
        self.explore_policy = self.policy
        self.draw_policy = policy.GreedyPolicy(environment.state_space, environment.action_space,self.qvalues)
        
    def update(self, state, action, reward, next_state, done):
        pass
        
    def get_action(self, state):
        return self.get_explore_action(state)

    def get_policy_action(self, state):
        return self.policy.step(state)

    def get_explore_action(self, state):
        return self.explore_policy.step(state)

    def get_draw_policy_action(self, state):
        return self.draw_policy.step(state)
        

    def render(self, env, frame):

        for idx, qvalues in enumerate(self.qvalues):
            position = env.idx2state[idx]

            # Don'self.t draw on end or blocked positions
            if position in env.end_positions or position in env.blocked_positions:
                continue

            x, y = position

            # for each action in state cell
            for action, qvalue in enumerate(qvalues):

                tanh_qvalue = np.tanh(qvalue * 0.1)  # for vizualization only

                # draw (state, action) qvalue traingle

                if action == 0:
                    dx2, dy2, dx3, dy3, dqx, dqy = 0.0, 1.0, 1.0, 1.0, .5, .85
                if action == 1:
                    dx2, dy2, dx3, dy3, dqx, dqy = 1.0, 0.0, 1.0, 1.0, .85, .5
                if action == 2:
                    dx2, dy2, dx3, dy3, dqx, dqy = 0.0, 0.0, 1.0, 0.0, .5, .15
                if action == 3:
                    dx2, dy2, dx3, dy3, dqx, dqy = 0.0, 0.0, 0.0, 1.0, .15, .5

                p1 = env.pos_to_frame((x + 0.5, y + 0.5))
                p2 = env.pos_to_frame((x + dx2, y + dy2))
                p3 = env.pos_to_frame((x + dx3, y + dy3))

                # pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
                # pts = pts.reshape((-1, 1, 2))
                pts = np.array([list(p1), list(p2), list(p3)], np.int32)

                if tanh_qvalue > 0:
                    color = (0, int(tanh_qvalue * 255), 0)
                elif tanh_qvalue < 0:
                    color = (0, 0, -int(tanh_qvalue * 255))
                else: 
                    color = (0, 0, 0)

                cv2.fillPoly(frame, [pts], color)

                qtext = "{:5.2f}".format(qvalue)
                if qvalue > 0.0:
                    qtext = '+' + qtext
                env.text_to_frame(frame, qtext, (x+dqx, y+dqy), (255,255,255), 0.4, 1)

            # draw arrows indicating policy or best action
            draw_action = self.get_draw_policy_action(idx)

            if draw_action == 0:
                start, end  = (x+.5, y+.4), (x+.5, y+.6)

            elif draw_action == 1:
                start, end  = (x+.4, y+.5), (x+.6, y+.5)

            elif draw_action == 2:
                start, end  = (x+.5, y+.6), (x+.5, y+.4)

            elif draw_action == 3:
                start, end  = (x+.6, y+.5), (x+.4, y+.5)

            cv2.arrowedLine(frame, env.pos_to_frame(start), env.pos_to_frame(end), (255,155,155), 8, line_type=8, tipLength=0.9)
        
        return frame  
            

class TDAgent(BaseAgent):

    def update(self, state, action, reward, next_state, terminal):

        # Q(s,a) = (1.0 - alpha) * Q(s,a) + alpha * (reward + discount * V(s'))

        if terminal==True:
            qval_dash = reward
        else:
            qval_dash = reward + self.discount * self.get_next_value(next_state)
            
        qval_old = self.qvalues[state][action]      
        qval = (1.0 - self.alpha)* qval_old + self.alpha * qval_dash

        self.qvalues[state][action] = qval
        return terminal

    def get_next_value(self, state):
        pass




class MCValueEstimator(BaseAgent):
    def __init__(self, alpha, discount, env):
        super().__init__(alpha,    epsilon, discount, env)

        self.policy = policy.FixedRandomPolicy(env.state_space, env.action_space) 
        self.explore_policy = self.policy 
    
    def update(self, state, action, reward, next_state, next_action, done):
        qval_old = self.qvalues[state][action]      
        qval = reward + self.discount * ((1.0 - self.alpha)* qval_old + self.alpha * self.qvalues[next_state][action])
        self.qvalues[state][action] = qval


# -----------------------------------------------------
# ------------ First-Value MC Prediction ---------------
# -----------------------------------------------------

class FVMCPrediction(BaseAgent):

    def get_next_value(self, next_state):

        # estimate V(s) as maximum of Q(state,action) over possible actions
        value = self.qvalues[state][0]
       
        for action in range(self.action_space):
            q_val = self.qvalues[state][action]
            if q_val > value:
                value = q_val

        return value


# -----------------------------------------------------
# ------------ ---- Q-Learning Agent -----------------------
# -----------------------------------------------------

class QLearningAgent(TDAgent):

    def __init__(self, alpha,  discount, environment, epsilon=0.2):
        super().__init__(alpha, discount, environment)

        self.policy = policy.GreedyPolicy(environment.state_space, environment.action_space, self.qvalues)
        self.explore_policy = policy.EpsilonGreedyPolicy(environment.state_space, environment.action_space, self.qvalues, epsilon)

    def get_next_value(self, next_state):
        return self.qvalues[next_state][self.get_policy_action(next_state)]


# -----------------------------------------------------
# ------------------------- SARSA Agent -------------------------
# -----------------------------------------------------
class SARSAAgent(TDAgent):

    def __init__(self, alpha, discount, env, epsilon=0.2):
        super().__init__(alpha, discount, env)

        self.policy = policy.EpsilonGreedyPolicy(
            env.state_space, env.action_space, 
            self.qvalues, epsilon)
        self.explore_policy = self.policy

        self.next_action = self.get_policy_action(env.get_state())

    def get_action(self, state):
        # I already know my next action since I chose it in the update
        return self.next_action

    def update(self, state, action, reward, next_state, terminal):

        # choose and remember next action
        self.next_action = self.get_policy_action(next_state)
        
        return super().update(state, action, reward, next_state, terminal)

    def get_next_value(self, next_state):
        return self.qvalues[next_state][self.next_action]


# -----------------------------------------------------
# ------------------- self.n-Steo SARSA Agent ------------------
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
        print("initial state is", state)
        action = self.get_explore_action(state)
        print("taking action", action)
        self.actions[0] = action
        self.next_action = action

        self.T = np.infty
        self.t = 0
        self.tau = 0

    def get_action(self, state):
        return self.next_action

    def update(self, state, action, reward, next_state, done):
        if (self.tau < self.T - 1):

            print("self.t=",self.t, ", tau=", self.tau, "self.T=", self.T)
            print("states=",self.states)
            print("actions=",self.actions)
            print("rewards=",self.rewards)

            print("ended up in state", next_state,"with reward", reward )
            self.rewards[(self.t + 1) % self.n] = reward
            self.states[(self.t + 1) % self.n] = next_state

            print("states=",self.states)
            print("actions=",self.actions)
            print("rewards=",self.rewards)

            # We're in a terminal state, need to unroll the rewards
            if done == True:
                self.T = self.t + 1
            else:
                # Sample next state to use
                print("next state is", next_state)
                self.next_action = self.get_explore_action(next_state)
                print("next action is", self.next_action)
                self.actions[(self.t + 1) % self.n] = self.next_action
        
            tau, T, t, n = self.tau, self.T, self.t, self.n
            discount, alpha = self.discount, self.alpha
            states, actions, qvalues = self.states, self.actions, self.qvalues

            tau = t - n + 1
            if tau >= 0:  # We have at least n observations
                print("updating Q values")
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


    def get_next_value(self, next_state, next_action):
        return self.qvalues[next_state][next_action]





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
