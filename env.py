import numpy as np
import random

class RubiksCube():

    def __init__(self):

        self.reward_for_solving = 20
        self.nS = 144
        self.nA = 6

        self.reset(0)

    # n is max number of moves
    def step(self, action: int, moves: int, n: int):
        done = False
        self.do_step(action)
        reward = -1


        if moves >= n:
            done = True
        
        if self.is_solved():
            reward = self.reward_for_solving
            done = True

        return self.flatten(), reward, done, {}

    def do_step(self, action):

        if action == 0:
            self.step_1(self.red)
            self.step_1(self.white)
            self.step_1(self.orange)
            self.step_1(self.yellow)
            self.step_1(self.blue)
            self.step_1(self.green)

        elif action == 1:
            self.step_2(self.red)
            self.step_2(self.white)
            self.step_2(self.orange)
            self.step_2(self.yellow)
            self.step_2(self.blue)
            self.step_2(self.green)

        elif action == 2:
            self.step_3(self.red)
            self.step_3(self.white)
            self.step_3(self.orange)
            self.step_3(self.yellow)
            self.step_3(self.blue)
            self.step_3(self.green)

        elif action == 3:
            self.step_4(self.red)
            self.step_4(self.white)
            self.step_4(self.orange)
            self.step_4(self.yellow)
            self.step_4(self.blue)
            self.step_4(self.green)

        elif action == 4:
            self.step_5(self.red)
            self.step_5(self.white)
            self.step_5(self.orange)
            self.step_5(self.yellow)
            self.step_5(self.blue)
            self.step_5(self.green)

        elif action == 5:
            self.step_6(self.red)
            self.step_6(self.white)
            self.step_6(self.orange)
            self.step_6(self.yellow)
            self.step_6(self.blue)
            self.step_6(self.green)

    # facing front, flip right side backwards
    def step_1(self, state):
        
        state[:, 1] = np.concatenate((state[6:8, 1], state[:6, 1], state[8:, 1]))
        state[8:10] = np.vstack((np.array([state[9, 0], state[8, 0]]), np.array([state[9, 1], state[8, 1]])))

    def step_2(self, state):
        state[:, 1] = np.concatenate((state[2:8, 1], state[:2, 1], state[8:, 1]))
        state[8:10] = np.vstack((state[8:10, 1], state[8:10, 0]))

    def step_3(self, state):
        new_state = np.vstack((state[8:10], self.rotation_cw(state[2:4]), self.rotation_twice(state[10:]), self.rotation_ccw(state[6:8]), self.rotation_twice(state[4:6]), state[:2]))
        self.step_1(new_state)
        state[:, :] = np.vstack((new_state[10:], self.rotation_ccw(new_state[2:4]), self.rotation_twice(new_state[8:10]), self.rotation_cw(new_state[6:8]), new_state[:2], self.rotation_twice(new_state[4:6])))

    def step_4(self, state):
        new_state = np.vstack((state[8:10], self.rotation_cw(state[2:4]), self.rotation_twice(state[10:]), self.rotation_ccw(state[6:8]), self.rotation_twice(state[4:6]), state[:2]))
        self.step_2(new_state)
        state[:, :] = np.vstack((new_state[10:], self.rotation_ccw(new_state[2:4]), self.rotation_twice(new_state[8:10]), self.rotation_cw(new_state[6:8]), new_state[:2], self.rotation_twice(new_state[4:6])))

    def step_5(self, state):
        new_state = np.vstack((self.rotation_ccw(state[:2]), self.rotation_ccw(state[8:10]), self.rotation_cw(state[4:6]), self.rotation_ccw(state[10:]), self.rotation_ccw(state[6:8]), self.rotation_ccw(state[2:4])))
        self.step_1(new_state)
        state[:, :] = np.vstack((self.rotation_cw(new_state[:2]), self.rotation_cw(new_state[10:]), self.rotation_ccw(new_state[4:6]), self.rotation_cw(new_state[8:10]), self.rotation_cw(new_state[2:4]), self.rotation_cw(new_state[6:8])))
        
    def step_6(self, state):
        new_state = np.vstack((self.rotation_ccw(state[:2]), self.rotation_ccw(state[8:10]), self.rotation_cw(state[4:6]), self.rotation_ccw(state[10:]), self.rotation_ccw(state[6:8]), self.rotation_ccw(state[2:4])))
        self.step_2(new_state)
        state[:, :] = np.vstack((self.rotation_cw(new_state[:2]), self.rotation_cw(new_state[10:]), self.rotation_ccw(new_state[4:6]), self.rotation_cw(new_state[8:10]), self.rotation_cw(new_state[2:4]), self.rotation_cw(new_state[6:8])))
        
    ## rotation of side
    def rotation_cw(self, side):

        return np.vstack((np.array([side[1, 0], side[0, 0]]), np.array([side[1, 1], side[0, 1]])))

    def rotation_ccw(self, side):

        return np.vstack((side[:, 1], side[:, 0]))
    
    def rotation_twice(self, side):

        return self.rotation_cw(self.rotation_cw(side))
    
    def reset(self, n: int):

        self.numbered = np.vstack([ [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24]])

        self.red = np.vstack([   [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        self.white = np.vstack([ [0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        self.orange = np.vstack([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        self.yellow = np.vstack([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]])
        self.blue = np.vstack([  [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])
        self.green = np.vstack([ [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1]])

        self.solved_red = self.red.copy()
        self.solved_white = self.white.copy()
        self.solved_orange = self.orange.copy()
        self.solved_yellow = self.yellow.copy()
        self.solved_blue = self.blue.copy()
        self.solved_green = self.green.copy()

        mix_actions = self.mix(n)

        return self.flatten(), mix_actions

    def mix(self, num_actions):
        actions = []
        for _ in range(num_actions):

            action = self.sample_action()
            actions.append(action)

            self.do_step(action)
        return actions


    def sample_action(self):

        action = random.randint(0, self.nA - 1)

        return action
    
    def flatten(self):

        return np.concatenate((self.red.flatten(), self.white.flatten(), self.orange.flatten(), self.yellow.flatten(), self.blue.flatten(), self.green.flatten()))
    
    def is_solved(self):

        return np.all(self.red == self.solved_red) and np.all(self.white == self.solved_white) and np.all(self.orange == self.solved_orange) and np.all(self.yellow == self.solved_yellow) and np.all(self.blue == self.solved_blue) and np.all(self.green == self.solved_green)

