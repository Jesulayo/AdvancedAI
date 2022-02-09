import numpy as np

def grid_world_example(grid_size=(3, 4),
                       black_cells=[(1,1)],
                       white_cell_reward=-0.02,
                       green_cell_loc=(0,3),
                       red_cell_loc=(1,3),
                       green_cell_reward=1.0,
                       red_cell_reward=-1.0,
                       action_lrfb_prob=(.1, .1, .8, 0.),
                       start_loc=(0, 0)
                      ):
    num_states = grid_size[0] * grid_size[1]
    num_actions = 4
    P = np.zeros((num_actions, num_states, num_states))
    R = np.zeros((num_states, num_actions))

    def fill_in_probs():
        # helpers
        to_2d = lambda x: np.unravel_index(x, grid_size)
        to_1d = lambda x: np.ravel_multi_index(x, grid_size)

        def hit_wall(cell):
            if cell in black_cells:
                return True
            try: # ...good enough...
                to_1d(cell)
            except ValueError as e:
                return True
            return False

        # make probs for each action
        a_up = [action_lrfb_prob[i] for i in (0, 1, 2, 3)]
        a_down = [action_lrfb_prob[i] for i in (1, 0, 3, 2)]
        a_left = [action_lrfb_prob[i] for i in (2, 3, 1, 0)]
        a_right = [action_lrfb_prob[i] for i in (3, 2, 0, 1)]
        actions = [a_up, a_down, a_left, a_right]
        for i, a in enumerate(actions):
            actions[i] = {'up':a[2], 'down':a[3], 'left':a[0], 'right':a[1]}

        # work in terms of the 2d grid representation

        def update_P_and_R(cell, new_cell, a_index, a_prob):
            if cell == green_cell_loc:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = green_cell_reward

            elif cell == red_cell_loc:
                P[a_index, to_1d(cell), to_1d(cell)] = 1.0
                R[to_1d(cell), a_index] = red_cell_reward

            elif hit_wall(new_cell):  # add prob to current cell
                P[a_index, to_1d(cell), to_1d(cell)] += a_prob
                R[to_1d(cell), a_index] = white_cell_reward

            else:
                P[a_index, to_1d(cell), to_1d(new_cell)] = a_prob
                R[to_1d(cell), a_index] = white_cell_reward

        for a_index, action in enumerate(actions):
            for cell in np.ndindex(grid_size):
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

    return P, R

if __name__ == "__main__":
    grid_world_example()