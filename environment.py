import numpy as np
import cv2
import sys
import json

ENV_JSON_FILE = "envs.json"

class Environment(object):
    """
    Grid World Environment.
    States are indexed (X, Y), starting from the bottom left corner, like in Russel & Norvig

    They are flattened bottom-left, going right, and up, i.e.:
    (0,1)=3, (1,0)=4, (2,0)=5
    (0,0)=0, (1,0)=1, (2,0)=2

    Actions are indexed from 0 (North) and then clockwise
    """

    def __init__(self, env_name, start_position=None, scale=100):

        # Initialize grid parameters from JSON file
        with open(ENV_JSON_FILE) as f:
            envs = json.load(f)
            for k,v in envs[env_name].items():
                 vars(self)[k] = v 

        self.end_positions = [tuple(l) for l in self.end_positions]
        self.blocked_positions = [tuple(l) for l in self.blocked_positions]

        # Non-JSON grid parameters
        self.state_space = self.gridW * self.gridH
        self.action_space = 4
        self.start_position = start_position
        self.scale = scale

        # Start state
        if self.start_position is None:
            self.position = self.init_random_start_state()
        else:
            self.position = self.start_position

        # Initialize quick-lookup lists 
        self.init_lookup_lists()

        # Generate fixed background render frame
        self.render_background()

    def init_lookup_lists(self):
        """
        Helper function to initialize quick lookup arrays
        """   
        self.state2idx = {}
        self.idx2state = {}
        self.idx2reward = {}
        for y in range(self.gridH):
            for x in range(self.gridW):
                idx = y * self.gridW + x
                self.state2idx[(x, y)] = idx
                self.idx2state[idx] = (x, y)
                self.idx2reward[idx] = self.default_reward
        for position, reward in zip(self.end_positions, self.end_rewards):
            self.idx2reward[self.state2idx[position]] = reward

    # Helper functions for rendering
    def pos_to_frame(self, pos):
        """
        Convert grid position to UI frame
            :param pos: 2-tuple to convert from gridworld to frame
        """   
        return (int((pos[0] + 0.0) * self.scale), int((self.gridH - pos[1] + 0.0) * self.scale))

    def text_to_frame(self, frame, text, pos, color=(255, 255, 255), fontscale=1, thickness=2):
        """
        Put text at grid position to UI frame coordinates
            :param frame: frame to draw on
            :paran text: text to write
            :param pos: 2-tuple of gridworld position (can be fraction)
        """   
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), _ = cv2.getTextSize(text, font, fontscale, thickness)
        textpos = (int((pos[0] + 0.0) * self.scale - w / 2), int((self.gridH - pos[1] + 0.0) * self.scale + h / 2))
        cv2.putText(frame, text, textpos, font, fontscale, color, thickness, cv2.LINE_AA)

    # Rendering the fixed background (happens once)
    def render_background(self):
        
        self.frame = np.zeros((self.gridH * self.scale, self.gridW * self.scale, 3), np.uint8)

        for position in self.blocked_positions:
            x, y = position
            cv2.rectangle(self.frame, self.pos_to_frame((x, y)), self.pos_to_frame((x + 1, y + 1)), (100, 100, 100), -1)

        for position, reward in zip(self.end_positions, self.end_rewards):
            text = str(int(reward))
            if reward > 0.0:
                text = '+' + text
            if reward >= 0.0:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            x, y = position
            self.text_to_frame(self.frame, text, (x + .5, y + .5), color)

    # Picking a random start position
    def init_random_start_state(self):

        while True:
            preposition = (np.random.choice(self.gridW), np.random.choice(self.gridH))

            if preposition not in self.end_positions and preposition not in self.blocked_positions:
                return preposition

    def get_state(self):
        return self.state2idx[self.position]

    # Main step generating new state and reward from current state and action
    def step(self, action, error=0.0):

        if action >= self.action_space:
            return

        # Action may be distorted with probability error
        if error > 0.0 and np.random.random() < error:
            wrong_action = np.random.choice(4)
            # print (f"Changing action {action} -> {wrong_action}")
            action = wrong_action

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

        if x_within and y_within and free and not terminal:
            self.position = proposed

        next_state = self.state2idx[self.position]
        reward = self.idx2reward[next_state]

        if terminal:  
            reward = 0

        return next_state, reward, terminal

    def reset_state(self):
        if self.start_position is None:
            self.position = self.init_random_start_state()
        else:
            self.position = self.start_position

    # Render agent and post-production (gridlines and state)
    def render(self, agent):

        frame = self.frame.copy()
        frame = agent.render(self, frame)

        # draw crossed lines
        # for i in range(self.state_space):
            # x, y = self.idx2state[i]

            # cv2.line(frame, self.pos_to_frame((x,y)), self.pos_to_frame((x+1,y+1)), (255, 255, 255), 2)
            # cv2.line(frame, self.pos_to_frame((x+1,y)), self.pos_to_frame((x,y+1)), (255, 255, 255), 2)

        # draw horizontal lines
        for i in range(self.gridH+1):
            cv2.line(frame, (0, i*self.scale), (self.gridW * self.scale, i*self.scale), (255, 255, 255), 2)

        # draw vertical lines
        for i in range(self.gridW+1):
            cv2.line(frame, (i*self.scale, 0), (i*self.scale, self.gridH * self.scale), (255, 255, 255), 2)

        # draw agent position (state)
        x, y = self.position

        cv2.rectangle(frame, self.pos_to_frame((x+.3, y+.3)), self.pos_to_frame((x+.7, y+.7)), (255, 255, 0), 3)

        # render everything
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == 27: sys.exit()

