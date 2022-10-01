import random
import threading
import STcpClient
import time
import sys
import numpy as np


def bfs(map, pos, depth=999):
    '''
    map: 
        0: empty
        1: power (highest priority)
        2: bomb or ghost
        3: pellet
    '''
    visited = []
    unvisited = [[pos, []]]
    while unvisited:
        (y, x), act_list = unvisited.pop(0)
        # improvable: 加入reward，值到一個夠高的reward才return
        visited.append((y, x))
        if 1 in map:
            if map[y, x] == 1 and len(act_list) >= 1 and (y, x) != (0, 0):
                return act_list[0]

        elif map[y, x] == 3 and len(act_list) >= 1:
            return act_list[0]
    
        if map[y, x] == 2:
            continue

        for act in legal_moves[y][x]:
            if act == 0 and (y, x - 1) not in visited:
                unvisited.append([(y, x - 1), act_list + [0]])
            if act == 1 and (y, x + 1) not in visited:
                unvisited.append([(y, x + 1), act_list + [1]])
            if act == 2 and (y - 1, x) not in visited:
                unvisited.append([(y - 1, x), act_list + [2]])
            if act == 3 and (y + 1, x) not in visited:
                unvisited.append([(y + 1, x), act_list + [3]])
        
        # greedy 根據 reward sort 優先展開 reward 高的路徑
    return random.choice([1, 2, 3, 4])

class MyThread(threading.Thread): 
   def __init__(self, *args, **keywords): 
       threading.Thread.__init__(self, *args, **keywords) 
       self.killed = False      
   def start(self):         
       self.__run_backup = self.run         
       self.run = self.__run                
       threading.Thread.start(self)         
   def __run(self):         
       sys.settrace(self.globaltrace)         
       self.__run_backup()         
       self.run = self.__run_backup         
   def globaltrace(self, frame, event, arg):         
       if event == 'call':             
           return self.localtrace         
       else:             
           return None        
   def localtrace(self, frame, event, arg):         
       if self.killed:             
          if event == 'line':                 
              raise SystemExit()         
       return self.localtrace         
   def kill(self):         
       self.killed = True

def getStep(playerStat, ghostStat, propsStat):
    global action
    '''
    control of your player
    0: left, 1:right, 2: up, 3: down 4:no control
    format is (control, set landmine or not) = (0~3, True or False)
    put your control in action and time limit is 0.04sec for one step
    '''
    state = np.zeros((16, 16))

    '''
    1: 優先找的
    2: 不能碰到的
    3: 沒有 1 的時候找的
    4: 地雷
    '''


    for type, x_org, y_org in propsStat:
        x = x_org // 25
        y = y_org // 25
        dx = x_org - x * 25
        dy = y_org - y * 25
        if type == 1:
            state[y, x] = 1
        elif type == 2:
            state[y, x] = 3
        elif type == 3:
            # improveable: 判斷炸彈的位置
            if dx > 11 and dx < 15:
                # bomb is put alone y-axis
                state[min(y + 1, 15), x] = 2
            
            if dy > 11 and dy < 15:
                # bomb is put alone x-axis
                state[y, min(x + 1, 15)] = 2

            state[y, x] = 2
        elif type == 0:
            state[y, x] = 4


    for x, y in ghostStat:
        x = x // 25
        y = y // 25
        if playerStat[3] > 0:
            state[y, x] = 1
        else:
            state[y, x] = 2
            # improveable: 鬼的周圍都要避開

            if 0 in legal_moves[y][x]:
                state[max(y - 1, 0), x] = 2
            if 1 in legal_moves[y][x]:
                state[min(y + 1, 15), x] = 2
            if 2 in legal_moves[y][x]:
                state[y, max(x - 1, 0)] = 2
            if 3 in legal_moves[y][x]:
                state[y, min(x + 1, 15)] = 2

    x, y = playerStat[0], playerStat[1]
    x = x // 25
    y = y // 25
    state[y, x] = 9
    # print(state)
    move = int(bfs(state, (y, x)))
    landmine = False
    if playerStat[2] > 0:
        landmine = random.choice([True, False])
    action = [move, landmine]



# props img size => pellet = 5*5, landmine = 11*11, bomb = 11*11
# player, ghost img size=23x23


if __name__ == "__main__":
    # parallel_wall = zeros([16, 17])
    # vertical_wall = zeros([17, 16])
    (stop_program, id_package, parallel_wall, vertical_wall) = STcpClient.GetMap()
    legal_moves = [] 
    '''
    [
        [[0, 1], [0, 1, 2], [1, 2, 3] , ...]
        [[0, 1], [0, 1, 2], [1, 2, 3] , ...]
        [[0, 1], [0, 1, 2], [1, 2, 3] , ...]
    ]

    0: left, 1:right, 2:up, 3:down
    '''
    for y in range(16):
        row = []
        for x in range(16):
            legal = []
            if vertical_wall[x, y] == 0:
                legal.append(0)
            if vertical_wall[x + 1, y] == 0:
                legal.append(1)
            if parallel_wall[x, y] == 0:
                legal.append(2)
            if parallel_wall[x, y + 1] == 0:
                legal.append(3)
            row.append(legal)
        legal_moves.append(row)
    memory = []
    while True:
        # playerStat: [x, y, n_landmine,super_time]
        # otherplayerStat: [x, y, n_landmine, super_time]
        # ghostStat: [[x, y],[x, y],[x, y],[x, y]]
        # propsStat: [[type, x, y] * N]
        (stop_program, id_package, playerStat,otherPlayerStat, ghostStat, propsStat) = STcpClient.GetGameStat()
        
        if stop_program:
            break
        elif stop_program is None:
            break
        global action
        action = None

        user_thread = MyThread(target=getStep, args=(playerStat, ghostStat, propsStat))
        user_thread.start()

        time.sleep(4/100)
        if action == None:
            user_thread.kill()
            user_thread.join()
            action = [4, False]

        if len(memory) > 6:
            
            if memory[0] == memory[5]:
                a = legal_moves[playerStat[1]//25][playerStat[0]//25]
                action[0] = random.choice(a)
            memory = []
        else:
            memory.append(playerStat[:2])

        is_connect=STcpClient.SendStep(id_package, action[0], action[1])
        if not is_connect:
            break


