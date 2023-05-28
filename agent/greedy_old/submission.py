import copy
import math
import numpy as np

def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def my_controller(observation_list, action_space_list, is_act_continuous, eval=False):
    # joint_action = []
    # obs = observation_list.copy()
    # width = obs['board_width']
    # height = obs['board_height']
    # o_index = obs['controlled_snake_index']
    # state = np.zeros((height, width))

    state_copy = observation_list.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    o_index = state_copy['controlled_snake_index']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list
    # beans = obs[1]
    # snakes = [[], []]
    # for i in beans:
    #     state[i[0], i[1]] = 1
    # for l in [2, 3, 4, 5, 6, 7]:
    #     if l != o_index:
    #         for i in obs[l]:
    #             state[i[0], i[1]] = 3
    #         snakes[1] += obs[l]
    # for i in obs[o_index]:
    #     state[i[0], i[1]] = 2
    # snakes[0] += obs[o_index]
    # state[snakes[0][-1][0]][snakes[0][-1][1]]=0.
    # state[snakes[1][-1][0]][snakes[1][-1][1]]=0.
    # print(state)
    actions = greedy_snake(state, beans, snakes, board_width, board_height, [o_index - 2])
    # print(actions)
    # player = []
    if not eval:
        each = [0] * 4
        each[actions[0]] = 1
        # player.append(each)
        # joint_action.append(player)
        # return joint_action
        # print(each)
        return [each]
    else:
        return actions[0]

def evaluation(states):
    actions = []
    for state in states:
        actions.append(my_controller(state, [], [], eval=True))
    return actions


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def get_actions(head_x,head_y,bean_x,bean_y,head_surrounding,width,height,actions,state,flag=1):
    next_distances = []
    surrounding=[[(head_y - 1) % height,head_x],[(head_y + 1) % height,head_x],
                 [head_y,(head_x - 1) % width],[head_y,(head_x + 1) % width]]
    if flag:
        regi=[]
        for bean in surrounding:
            count=get_region_size(state,width,height,bean[1],bean[0])
            if count==0:
                regi.append(math.inf)
            elif count<=4.0:
                regi.append(1000/np.sqrt(count))
            else:
                regi.append(0.0)
    # print("——————————————————————————————————————")
    # print(state)
    # print(head_y,head_x)
    # print(regi)
    up_distance = math.inf if head_surrounding[0] > 1 else \
        min(abs(head_x - bean_x), abs(width - abs(head_x - bean_x))) + \
        min(abs((head_y - 1) % height - bean_y), abs(height - abs((head_y - 1) % height - bean_y)))
    next_distances.append(up_distance)

    down_distance = math.inf if head_surrounding[1] > 1 else \
        min(abs(head_x - bean_x), abs(width - abs(head_x - bean_x))) +\
        min(abs((head_y + 1) % height - bean_y), abs(height - abs((head_y + 1) % height - bean_y)))
    next_distances.append(down_distance)

    left_distance = math.inf if head_surrounding[2] > 1 else \
        min(abs((head_x - 1) % width - bean_x), abs(width - abs(abs((head_x - 1) % width - bean_x)))) +\
        min(abs(head_y - bean_y), abs(height - abs(head_y - bean_y)))
    next_distances.append(left_distance)

    right_distance = math.inf if head_surrounding[3] > 1 else \
        min(abs((head_x + 1) % width - bean_x), abs(width - abs(abs((head_x + 1) % width - bean_x)))) + \
        min(abs(head_y - bean_y), abs(height - abs(head_y - bean_y)))
    next_distances.append(right_distance)
    if flag:
        next_distances = list(np.array(next_distances)+np.array(regi))
    # print(next_distances)
    actions.append(next_distances.index(min(next_distances)))


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map




def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index

def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions








def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions[i]
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action