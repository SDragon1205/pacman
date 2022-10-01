import random
import threading
import STcpClient
import time
import sys
import numpy as np
import pickle
from collections import deque

class nn:
    def __init__(self, input_shape, hidden_neurons, output_shape, learning_rate):
        self.l1_weights = np.random.normal(scale=0.1, size=(hidden_neurons, hidden_neurons))
        self.l1_biases = np.zeros(hidden_neurons)

        self.l2_weights =  np.random.normal(scale=0.1, size=(hidden_neurons, output_shape))
        self.l2_biases = np.zeros(output_shape)

        self.learning_rate = learning_rate


    def linear(x, derivation=False):
        if derivation:
            return 1
        else:
            return x

    def relu(x, derivation=False):
        if derivation:
            return 1.0 * (x > 0)
        else:
            return np.maximum(x, 0)


    def fit(self, x, y, epochs=1):
        """
        method implements backpropagation
        """
        for _ in range(epochs):
        # Forward propagation
            # First layer
            u1 = np.dot(x, self.l1_weights) + self.l1_biases
            l1o = self.relu(u1)

            # Second layer
            u2 = np.dot(l1o, self.l2_weights) + self.l2_biases
            l2o = self.linear(u2)

        # Backward Propagation
            # Second layer
            d_l2o = l2o - y
            d_u2 = self.linear(u2, derivation=True)

            g_l2 = np.dot(l1o.T, d_u2 * d_l2o)
            d_l2b = d_l2o * d_u2
            # First layer
            d_l1o = np.dot(d_l2o , self.l2_weights.T)
            d_u1 = self.relu(u1, derivation=True)

            g_l1 = np.dot(x.T, d_u1 * d_l1o)
            d_l1b = d_l1o * d_u1

        # Update weights and biases
            self.l1_weights -= self.learning_rate * g_l1
            self.l1_biases -= self.learning_rate * d_l1b.sum(axis=0)

            self.l2_weights -= self.learning_rate * g_l2
            self.l2_biases -= self.learning_rate * d_l2b.sum(axis=0)

        # Return actual loss
        return np.mean(np.subtract(y, l2o)**2)

    def predict(self, x):
        """
        method predicts q-values for state x
        """
        #First layer
        u1 = np.dot(x, self.l1_weights) + self.l1_biases
        l1o = self.relu(u1)

        #Second layer
        u2 = np.dot(l1o, self.l2_weights) + self.l2_biases
        l2o = self.linear(u2)

        return l2o

    def save_model(self, name):
        """
        method saves model
        """
        with open("{}.pkl" .format(name), "wb") as model:
            pickle.dump(self, model, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        """
        method loads model
        """
        with open("{}".format(name), "rb") as model:
            tmp_model = pickle.load(model)

        self.l1_weights = tmp_model.l1_weights
        self.l1_biases = tmp_model.l1_biases

        self.l2_weights =  tmp_model.l2_weights
        self.l2_biases = tmp_model.l2_biases

        self.learning_rate = tmp_model.learning_rate


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

def getStep(state, playerStat):
    global action
    '''
    control of your player
    0: left, 1:right, 2: up, 3: down 4:no control
    format is (control, set landmine or not) = (0~3, True or False)
    put your control in action and time limit is 0.04sec for one step
    '''


    
    move = random.choice([0, 1, 2, 3, 4])
    landmine = False
    if playerStat[2] > 0:
        landmine = random.choice([True, False])
    action = [move, landmine]

def train(model, memory, minibatch_size, gamma):
    if minibatch_size > len(memory):
        return
    minibatch = random.sample(memory, minibatch_size)

    state = np.array([i[0] for i in minibatch])
    action = [i[1] for i in minibatch]
    reward = [i[2] for i in minibatch]
    next_state = np.array([i[3] for i in minibatch])
    done = [i[4] for i in minibatch]

    q_value = model.predict(np.array(state))
    ns_model_pred = model.predict(np.array(next_state))

    for i in range(0, minibatch_size):
        if done[i] == 1:
            q_value[i][action[i]] = reward[i]
        else:
            q_value[i][action[i]] = reward[i] + gamma * np.max(ns_model_pred[i])

    model.fit(state, q_value)
    model.save_model("model")

# props img size => pellet = 5*5, landmine = 11*11, bomb = 11*11
# player, ghost img size=23x23


if __name__ == "__main__":
    # parallel_wall = zeros([16, 17])
    # vertical_wall = zeros([17, 16])
    global action
    (stop_program, id_package, parallel_wall, vertical_wall) = STcpClient.GetMap()
    next_state = None
    state = None
    action = None
    model = nn(16, 16, 4, 0.01)
    memory = []
    round = 0
    while True:
        # playerStat: [x, y, n_landmine,super_time]
        # otherplayerStat: [x, y, n_landmine, super_time]
        # ghostStat: [[x, y],[x, y],[x, y],[x, y]]
        # propsStat: [[type, x, y] * N]
        (stop_program, id_package, playerStat, otherPlayerStat, ghostStat, propsStat) = STcpClient.GetGameStat()
        if stop_program:
            break
        elif stop_program is None:
            break

        '''
        state:
            0: empty
            1: me, 2: other player, 3: ghost
            4: landmine, 5: power, 6: pellet, 7: bomb
        '''

        state = next_state
        next_state = np.zeros((16, 16)) # state[y, x]
        for item in propsStat:
            next_state[item[0]//25, item[1]//25] = item[0] + 4
        
        memory.append((state, action, reward, next_state, round))
        train(model, memory, 64, 0.9)


        action = None
        user_thread = MyThread(target=getStep, args=(next_state, playerStat))
        user_thread.start()
        time.sleep(4/100)
        if action == None:
            user_thread.kill()
            user_thread.join()
            action = [4, False]
        is_connect=STcpClient.SendStep(id_package, action[0], action[1])

        if not is_connect:
            break

def valid_move(x, y, parallel_wall, vertical_wall):
    valid_list = []
    if parallel_wall[y]:
        valid_list.append(3)
    if parallel_wall[y + 1]:
        valid_list.append(4)
    if vertical_wall[x]:
        valid_list.append(1)
    if vertical_wall[x + 1]:
        valid_list.append(2)

    return valid_list



