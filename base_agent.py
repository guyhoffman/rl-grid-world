import numpy as np
import policy
import cv2

# -----------------------------------------------------
# -------------------------- Base Agent ----------------
# -----------------------------------------------------

class BaseAgent:
    
    def __init__(self, alpha, discount, environment):
 
        self.alpha = alpha
        self.discount = discount

        ssp = environment.state_space
        asp = environment.action_space

        self.action_space = asp
        self.qvalues = np.zeros((ssp, asp), np.float32)

        self.optimal_policy = policy.RandomPolicy(ssp, asp)
        self.explore_policy = self.optimal_policy
        self.draw_policy = policy.GreedyPolicy(ssp, asp, self.qvalues)
        
    # def update(self, state, action, reward, next_state, done):
    #     pass
        
    def get_action(self, state):
        return self.get_explore_action(state)

    def get_optimal_action(self, state):
        return self.optimal_policy.step(state)

    def get_explore_action(self, state):
        return self.explore_policy.step(state)

    def get_draw_action(self, state):
        return self.draw_policy.step(state)        

    def render(self, env, frame):

        for idx, qvalues in enumerate(self.qvalues):
            position = env.idx2state[idx]

            # Don't draw on end or blocked positions
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

                pts = np.array([list(p1), list(p2), list(p3)], np.int32)

                if tanh_qvalue > 0:
                    color = (0, int(tanh_qvalue * 255), 0)
                elif tanh_qvalue < 0:
                    color = (0, 0, -int(tanh_qvalue * 255))
                else: 
                    color = (0, 0, 0)

                cv2.fillPoly(frame, [pts], color)

                # Draw Q-value text
                qtext = "{:5.2f}".format(qvalue)
                if qvalue > 0.0:
                    qtext = '+' + qtext
                env.text_to_frame(frame, qtext, (x+dqx, y+dqy), (255,255,255), 0.4, 1)

            # draw arrows indicating policy or best action
            draw_action = self.get_draw_action(idx)

            if draw_action == 0:
                start, end  = (x+.5, y+.4), (x+.5, y+.6)

            elif draw_action == 1:
                start, end  = (x+.4, y+.5), (x+.6, y+.5)

            elif draw_action == 2:
                start, end  = (x+.5, y+.6), (x+.5, y+.4)

            elif draw_action == 3:
                start, end  = (x+.6, y+.5), (x+.4, y+.5)

            cv2.arrowedLine(frame, env.pos_to_frame(start), env.pos_to_frame(end), (255,225,225), 5, line_type=8, tipLength=0.5)
        
        return frame  
