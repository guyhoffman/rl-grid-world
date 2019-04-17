import numpy as np
import cv2
import sys


class Environment(object):
    """
    Grid World Environment.
    States are indexed (X, Y), starting from the bottom left corner, like in Russel & Norvig

    They are flattened bottom-left, going right, and up, i.e.:
    (0,1)=3, (1,0)=4, (2,0)=5
    (0,0)=0, (1,0)=1, (2,0)=2

    Actions are indexed from 0 (North) and then clockwise
    """

    def __init__(self, gridW, gridH, end_positions, end_rewards, blocked_positions, start_position, default_reward, scale=100):

        self.action_space = 4
        self.state_space = gridH * gridW
        self.gridH = gridH
        self.gridW = gridW
        self.scale = scale

        self.end_positions = end_positions
        self.end_rewards = end_rewards
        self.blocked_positions = blocked_positions

        self.start_position = start_position
        if self.start_position is None:
            self.position = self.init_start_state()
        else:
            self.position = self.start_position

        self.state2idx = {}
        self.idx2state = {}
        self.idx2reward = {}
        for y in range(self.gridH):
            for x in range(self.gridW):
                idx = y * self.gridW + x
                self.state2idx[(x, y)] = idx
                self.idx2state[idx] = (x, y)
                self.idx2reward[idx] = default_reward
        for position, reward in zip(self.end_positions, self.end_rewards):
            self.idx2reward[self.state2idx[position]] = reward

        self.frame = np.zeros((self.gridH * self.scale, self.gridW * self.scale, 3), np.uint8)

        for position in self.blocked_positions:
            x, y = position
            cv2.rectangle(self.frame, self.pos_to_frame((x, y)), self.pos_to_frame((x + 1, y + 1)), (100, 100, 100), -1)

        for position, reward in zip(self.end_positions, self.end_rewards):
            text = str(int(reward))
            if reward > 0.0:
                text = '+' + text
            if reward > 0.0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            x, y = position
            self.text_to_frame(self.frame, text, (x + .5, y + .5), color)

    def pos_to_frame(self, pos):
        return (int((pos[0] + 0.0) * self.scale), int((self.gridH - pos[1] + 0.0) * self.scale))

    def text_to_frame(self, frame, text, pos, color=(255, 255, 255), fontscale=1, thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(text, font, fontscale, thickness)
        textpos = (int((pos[0] + 0.0) * self.scale - w / 2), int((self.gridH - pos[1] + 0.0) * self.scale + h / 2))
        cv2.putText(frame, text, textpos, font, fontscale, color, thickness, cv2.LINE_AA)

    def init_start_state(self):

        while True:

            preposition = (np.random.choice(self.gridW), np.random.choice(self.gridH))

            if preposition not in self.end_positions and preposition not in self.blocked_positions:

                return preposition

    def get_state(self):
        return self.state2idx[self.position]

    def step(self, action):

        if action >= self.action_space:
            return

        if action == 0:  # North
            proposed = (self.position[0], self.position[1] + 1)

        elif action == 1:  # East
            proposed = (self.position[0] + 1, self.position[1])

        elif action == 2:  # South
            proposed = (self.position[0], self.position[1] - 1)

        elif action == 3:  # West
            proposed = (self.position[0] - 1, self.position[1])

        x_within = proposed[0] >= 0 and proposed[0] < self.gridW
        y_within = proposed[1] >= 0 and proposed[1] < self.gridH
        free = proposed not in self.blocked_positions
        terminal =  self.position in self.end_positions

        print ("Should move from", self.position, "to", proposed,"?")
        if x_within and y_within and free and not terminal:
            print ("Yes")
            self.position = proposed

        next_state = self.state2idx[self.position]
        reward = self.idx2reward[next_state]

        if terminal:
            print ("I'm moving from terminal so")
            done = True
            reward = 0
        else:
            done = False

        return next_state, reward, done

    def reset_state(self):
        if self.start_position is None:
            self.position = self.init_start_state()
        else:
            self.position = self.start_position

    def render(self, agent):

        frame = self.frame.copy()

        # for each state cell
        # print (np.min(qvalues_matrix), '->', np.max(qvalues_matrix))

        for idx, qvalues in enumerate(agent.qvalues):
            position = self.idx2state[idx]

            if position in self.end_positions or position in self.blocked_positions:
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

                p1 = self.pos_to_frame((x + 0.5, y + 0.5))
                p2 = self.pos_to_frame((x + dx2, y + dy2))
                p3 = self.pos_to_frame((x + dx3, y + dy3))

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
                self.text_to_frame(frame, qtext, (x+dqx, y+dqy), (255,255,255), 0.4, 1)

            # draw crossed lines
            cv2.line(frame, self.pos_to_frame((x,y)), self.pos_to_frame((x+1,y+1)), (255, 255, 255), 2)
            cv2.line(frame, self.pos_to_frame((x+1,y)), self.pos_to_frame((x,y+1)), (255, 255, 255), 2)

            # draw arrows indicating policy or best action
            draw_action = agent.get_draw_policy_action(idx)

            if draw_action == 0:
                start, end  = (x+.5, y+.4), (x+.5, y+.6)

            elif draw_action == 1:
                start, end  = (x+.4, y+.5), (x+.6, y+.5)

            elif draw_action == 2:
                start, end  = (x+.5, y+.6), (x+.5, y+.4)

            elif draw_action == 3:
                start, end  = (x+.6, y+.5), (x+.4, y+.5)

            cv2.arrowedLine(frame, self.pos_to_frame(start), self.pos_to_frame(end), (255,155,155), 8, line_type=8, tipLength=0.9)

        # draw horizontal lines

        for i in range(self.gridH+1):
            cv2.line(frame, (0, i*self.scale), (self.gridW * self.scale, i*self.scale), (255, 255, 255), 2)

        # draw vertical lines

        for i in range(self.gridW+1):
            cv2.line(frame, (i*self.scale, 0), (i*self.scale, self.gridH * self.scale), (255, 255, 255), 2)

        # draw agent

        x, y = self.position

        # y1 = int((y + 0.3)*self.scale)
        # x1 = int((x + 0.3)*self.scale)
        # y2 = int((y + 0.7)*self.scale)
        # x2 = int((x + 0.7)*self.scale)

        cv2.rectangle(frame, self.pos_to_frame((x+.3, y+.3)), self.pos_to_frame((x+.7, y+.7)), (255, 255, 0), 3)

        cv2.imshow('frame', frame)
        # cv2.moveWindow('frame', 0, 0)
        key = cv2.waitKey(1)
        if key == 27: sys.exit()

