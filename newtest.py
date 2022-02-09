import numpy as np

class Table():
    grid_size=(3, 4)
    black_cells=[(1,1)]
    white_cell_reward=-0.02
    green_cell_loc=(0,3)
    red_cell_loc=(1,3)
    green_cell_reward=1.0
    red_cell_reward=-1.0
    action_lrfb_prob=(.1, .1, .8, 0.)
    start_loc=(0, 0)

    def __init__(self):
        # print()
        self.grid_world_example()
        # grid_world_example()
        
    def grid_world_example(self):
        num_states = self.grid_size[0] * self.grid_size[1]
        num_actions = 4
        self.P = np.zeros((num_actions, num_states, num_states))
        self.R = np.zeros((num_states, num_actions))
        self.fill_in_probs()

    def fill_in_probs(self):
        # helpers
        to_2d = lambda x: np.unravel_index(x, self.grid_size)
        to_1d = lambda x: np.ravel_multi_index(x, self.grid_size)
        print(to_1d)

        def hit_wall(cell):
            if cell in self.black_cells:
                return True
            try: # ...good enough...
                to_1d(cell)
            except ValueError as e:
                return True
            return False

        # make probs for each action
        a_up = [self.action_lrfb_prob[i] for i in (0, 1, 2, 3)]
        a_down = [self.action_lrfb_prob[i] for i in (1, 0, 3, 2)]
        a_left = [self.action_lrfb_prob[i] for i in (2, 3, 1, 0)]
        a_right = [self.action_lrfb_prob[i] for i in (3, 2, 0, 1)]
        actions = [a_up, a_down, a_left, a_right]
        for i, a in enumerate(actions):
            print('i', i)
            print('a', a)
            actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}
        print('up', a_up)
        print('down', a_down)
        print('left', a_left)
        print('right', a_right)
        print(actions)

        # work in terms of the 2d grid representation

        def update_P_and_R(cell, new_cell, a_index, a_prob):
            if cell == self.green_cell_loc:
                self.P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                self.R[to_1d(cell), a_index] = self.green_cell_reward

            elif cell == self.red_cell_loc:
                self.P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                self.R[to_1d(cell), a_index] = self.red_cell_reward

            elif hit_wall(new_cell):  # add prob to current cell
                self.P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                self.R[to_1d(cell), a_index] = self.white_cell_reward

            else:
                self.P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                self.R[to_1d(cell), a_index] = self.white_cell_reward

        for a_index, action in enumerate(actions):
            for cell in np.ndindex(self.grid_size):
                # up
                new_cell = (cell[0]-1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['up'])

                # down
                new_cell = (cell[0]+1, cell[1])
                update_P_and_R(cell, new_cell, a_index, action['down'])

                # left
                new_cell = (cell[0], cell[1]-1)
                update_P_and_R(cell, new_cell, a_index, action['left'])

                # right
                new_cell = (cell[0], cell[1]+1)
                update_P_and_R(cell, new_cell, a_index, action['right'])

        print('P',self.P)
        print('R', self.R)
        return self.P, self.R

if __name__ == "__main__":
    Table()