import random
import threading
import STcpClient
import time
import sys
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
###############################################################################
class DQN_Agent:
    #
    # Initializes attributes and constructs CNN model and target_model
    #
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        
        # Hyperparameters
        self.gamma = 1.0            # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.1      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.update_rate = 500     # Number of steps until updating the target network###############################################################################
        
        # Construct DQN models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    #
    # Constructs CNN
    #
    def _build_model(self):
        model = Sequential()
        
        # Conv Layers
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam())
        return model

    #
    # Stores experience in replay memory
    #
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #
    # Chooses action based on epsilon-greedy policy
    #
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])  # Returns action using policy

    #
    # Trains the model using randomly selected experiences in the replay memory
    #
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))
            else:
                target = reward
                
            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state)
            
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            
            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
            
    #
    # Loads a saved model
    #
    def load(self, name):
        self.model.load_weights(name)

    #
    # Saves parameters of a trained model
    #
    def save(self, name):
        self.model.save_weights(name)
###############################################################################

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

def getStep(playerStat, ghostStat, propsStat, state):
    global action
    '''
    control of your player
    0: left, 1:right, 2: up, 3: down 4:no control
    format is (control, set landmine or not) = (0~3, True or False)
    put your control in action and time limit is 0.04sec for one step
    '''
    move = agent.act(state)

    landmine = False
    if playerStat[2] > 0:
        landmine = random.choice([True, False])
    action = [move, landmine]

# props img size => pellet = 5*5, landmine = 11*11, bomb = 11*11, power = 11*11
# player, ghost img size=23x23
'''
state:
    0: empty 1:wall
    2: me, 3: other player, 4: ghost
    5: landmine, 6: power, 7: pellet, 8: bomb
'''
def draw_wall(p_wall, v_wall):
    wall = np.zeros((401, 401)) # state[y, x]
    for i in range(16):
        for j in range(17):
            if p_wall[i][j]:
                x = i*25
                y = j*25
                for z in range(0, 25):
                    wall[x+z][y] = 1
    for i in range(17):
        for j in range(16):
            if v_wall[i][j]:
                x = i*25
                y = j*25
                for z in range(0, 25):
                    wall[x][y+z] = 1
    return wall

def draw_action(mm, playerStat, otherPlayerStat, ghostStat, propsStat):
    for x in range(playerStat[0] - 11, playerStat[0] + 12):
        for y in range(playerStat[1] - 11, playerStat[1] + 12):
            mm[x, y] = 2
    for hero in otherPlayerStat:
        for x in range(hero[0] - 11, hero[0] + 12):
            for y in range(hero[1] - 11, hero[1] + 12):
                mm[x, y] = 3
    for ghost in ghostStat:
        for x in range(ghost[0] - 11, ghost[0] + 12):
            for y in range(ghost[1] - 11, ghost[1] + 12):
                mm[x, y] = 4
    for food in propsStat:
        if food[0] == 2:#pellet
            for x in range(food[1] - 2, food[1] + 3):
                for y in range(food[2] - 2, food[2] + 3):
                    mm[x, y] = food[0] + 5
        else:
            for x in range(food[1] - 5, food[1] + 6):
                for y in range(food[2] - 5, food[2] + 6):
                    mm[x, y] = food[0] + 5
    
    return mm


if __name__ == "__main__":
    # parallel_wall = zeros([16, 17])
    # vertical_wall = zeros([17, 16])
    (stop_program, id_package, parallel_wall, vertical_wall) = STcpClient.GetMap()
    mapmap = draw_wall(parallel_wall, vertical_wall)

###############################################################################

    state_size = (401, 401, 1)###############################################################################
    action_size = len(['0', '1', '2', '3', '4'])
    agent = DQN_Agent(state_size, action_size)
    #agent.load('models/pacman')

    batch_size = 8
    skip_start = 0  # MsPacman-v0 waits for 90 actions before the episode begins
    done = False
    time = 0
    total_reward = 0
    game_score = 0
    
    #for skip in range(skip_start): # skip the start of each game###############################################################################???????????????????????
    #    env.step(0)

    step_two = False
    update = True
###############################################################################
    global action
    action = None

    while True:
        # playerStat: [x, y, n_landmine,super_time, score] dead_time total_time
        # otherplayerStat: [x, y, n_landmine, super_time]
        # ghostStat: [[x, y],[x, y],[x, y],[x, y]]
        # propsStat: [[type, x, y] * N] -> N <= 76
        # MAX_PELLET = 64
        # MAX_LANDMINES = 8
        # MAX_POWER = 4
        # 0 = landmine, 1 = power, 2 = pellet, 3 = bomb
        
        (stop_program, id_package, playerStat, otherPlayerStat, ghostStat, propsStat) = STcpClient.GetGameStat()

        #next_state, reward, done, _ = env.step(action)
        #next_state = parallel_wall, vertical_wall + playerStat,otherPlayerStat, ghostStat, propsStat
        #reward = score
        #done = stop_program
        if step_two:
            #next_state, reward, done, _ = env.step(action)
            next_state = draw_action(mapmap, playerStat, otherPlayerStat, ghostStat, propsStat)
            next_score = playerStat[4]
            next_x = playerStat[0]
            next_y = playerStat[1]
            next_power = playerStat[3]
            #if playerStat[5] < 0:
            #    reward -= 30###############################################################################
            reward = next_score - now_score
            if now_power - next_power > 1:#have power and touch bomb
                reward -= 25
            elif playerStat[3] = 0 and abs(next_x - now_x) + abs(next_y - now_y) > 5:#no power and touch ghost
                reward -= 100

            now_score = next_score
            now_x = next_x
            now_y = next_y
            now_power = next_power
            done = stop_program
            
            # Store sequence in replay memory
            agent.remember(state, action[0], reward, next_state, done)
            
            state = next_state
            game_score += reward
            reward -= 1  # Punish behavior which does not accumulate reward
            total_reward += reward
            
            if done:
                
                #print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                #    .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
                print("game score: {}, reward: {}, time:{}"
                    .format(game_score, total_reward, time))

                if update:
                    agent.update_target_model()
                
                agent.save('models/pacman')############################################################################### save
                
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        else:
            state = draw_action(mapmap, playerStat, otherPlayerStat, ghostStat, propsStat)
            now_score = playerStat[4]
            now_x = playerStat[0]
            now_y = playerStat[1]
            now_power = playerStat[3]
            #total_time = playerStat[6]   # Counter for total number of steps taken
        
        if stop_program:
            break
        elif stop_program is None:
            break

###############################################################################

        #total_time += 1
        time += 1
        
        # Every update_rate timesteps we update the target network parameters
        if time % agent.update_rate == 0:
            agent.update_target_model()
            update = False

###############################################################################

        user_thread = MyThread(target=getStep, args=(playerStat, ghostStat, propsStat))
        user_thread.start()
        time.sleep(4/100)
        if action == None:
            user_thread.kill()
            user_thread.join()
            action = [4, False]
        is_connect=STcpClient.SendStep(id_package, action[0], action[1])
        if not is_connect:
            break
    
        step_two = true
