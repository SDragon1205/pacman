from distutils.log import error
import os
import STcpServer
import gameUI
import threading
from gameUI import *
import pygame as pg
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
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
        self.update_rate = 1000     # Number of steps until updating the target network###############################################################################
        
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
    for y in range(401):
        for x in range(401):
            print(int(wall[x][y]), end = " ")
        print(" ")
    print("\nwall\n")
    return wall

def draw_action(mm, playerStat, otherPlayerStat, ghostStat, propsStat):
    for x in range(playerStat[0] - 11, playerStat[0] + 12):#23*23
        for y in range(playerStat[1] - 11, playerStat[1] + 12):
            mm[x, y] = 2
    for hero in otherPlayerStat:
        for x in range(hero[0] - 11, hero[0] + 12):#23*23
            for y in range(hero[1] - 11, hero[1] + 12):
                mm[x, y] = 3
    for ghost in ghostStat:
        for x in range(ghost[0] - 11, ghost[0] + 12):#23*23
            for y in range(ghost[1] - 11, ghost[1] + 12):
                mm[x, y] = 4
    for food in propsStat:
        if food[0] == 2:#pellet
            for x in range(food[1] - 2, food[1] + 3):#5*5
                for y in range(food[2] - 2, food[2] + 3):
                    mm[x, y] = food[0] + 5
        else:
            for x in range(food[1] - 5, food[1] + 6):#11*11
                for y in range(food[2] - 5, food[2] + 6):
                    mm[x, y] = food[0] + 5
    for y in range(401):
        for x in range(401):
            print(int(mm[x][y]), end = "")
        print(" ")
    print("\n\n\n\n\n\n\n\n\n\n\n")
    return mm

###############################################################################

def main(total_time):
    team_id = []
    with open('./path.txt', 'r', encoding="utf-8") as f:
        for line in f.readlines():
            if line[-1] == '\n':
                team_id.append(line[:-1])
            else:
                team_id.append(line)
    idTeam1 = int(team_id[0])
    pathExe1 = team_id[1]

    idTeam2 = int(team_id[2])
    pathExe2 = team_id[3]

    idTeam3 = int(team_id[4])
    pathExe3 = team_id[5]

    idTeam4 = int(team_id[6])
    pathExe4 = team_id[7]

    (success, failId) = STcpServer.StartMatch(idTeam1, pathExe1, idTeam2, pathExe2, idTeam3, pathExe3, idTeam4, pathExe4)

    if(not success):
        print("connection fail, teamId:", failId)
    else:
        print("connect success, init game")
        # 16*16 16*16
        p_wall, v_wall = gameUI.createMap()

        for playerid in range(4):
            success = STcpServer.SendMap(playerid, p_wall, v_wall)
            if success != 0:
                print("init fail")

        tt = gamestart(p_wall, v_wall, total_time)
        return tt

def gamestart(p_wall, v_wall, total_time):
    screen = initialize()
    wall_positions = drawWall(p_wall, v_wall)
    level = Game(wall_positions)
    clock = pg.time.Clock()
    SCORE = 0
    wall_sprites, safe_place = level.setupWalls(SKYBLUE)

    hero_sprites = level.setPlayer(PAC_MAN)
    ghost_sprites = level.setGhost(GHOST)
    landmine_sprites = level.setLandmines(YELLOW, BLACK)
    power_sprites = level.setPower(RED, BLACK)
    pellet_sprites = level.setPellet(GREEN, BLACK)
    bomb_sprites = level.setBomb()
    leave = False

###############################################################################
    mapmap = draw_wall(p_wall, v_wall)

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
###############################################################################

    gameScore = 0
    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                leave = True
                #pg.quit()
        if (len(landmine_sprites) == 0 and len(power_sprites) == 0 and len(pellet_sprites) == 0 and leave == False) or STcpServer.idPackage > 6000:
            leave = True
            #pg.quit()
        if leave == True: 
            done = True
            #break
        STcpServer.idPackage += 1
        # get status
        ghosts= []
        for ghost in ghost_sprites:
            ghosts.append((ghost.rect.left, ghost.rect.top))
        heros = []
        for hero in hero_sprites:
            heros.append((hero.rect.left, hero.rect.top, hero.landmine, hero.super_time, hero.score))
        foods = [(1, 8, 9), (1, 7, 15)]
        # 0 = landmine, 1 = power, 2 = pellet, 3 = bomb
        for landmine in landmine_sprites:
            foods.append((0, landmine.rect.left, landmine.rect.top))
        for power in power_sprites:
            foods.append((1, power.rect.left, power.rect.top))
        for pellet in pellet_sprites:
            foods.append((2, pellet.rect.left, pellet.rect.top))
        for bomb in bomb_sprites:
            foods.append((3, bomb.rect.left, bomb.rect.top))
        #
###############################################################################
        #send
        player_action = [(0, True) for i in range(4)]
        threads = []
        def sendstatus(playerID, ghost, hero, food):
            success, action = STcpServer.Sendstatus(playerID, ghosts, heros, foods)
            if success == 0:
                player_action[playerID] = action
            if success > 0 :
                print("random for player:", playerID)
                player_action[playerID] = (random.choice([0, 1, 2, 3]), random.choice([True, False]))

        # creat thread for each player
        for playerid in range(4):
            threads.append(threading.Thread(target = sendstatus, args = (playerid, ghosts, heros, foods)))
            threads[playerid].start()
            #threads[playerid].join()

        # wait for each thread end
        for playerid in range(4):
            threads[playerid].join()

###############################################################################

        playerStat = heros[0]
        otherPlayerStat = [heros[1], heros[2], heros[3]]
        ghostStat = ghosts
        propsStat = foods

        #update = True
        step_two = False

        if step_two:
            #next_state, reward, done, _ = env.step(action)
            next_state = draw_action(mapmap, playerStat, otherPlayerStat, ghostStat, propsStat)
            next_x = playerStat[0]
            next_y = playerStat[1]
            next_power = playerStat[3]
            next_score = playerStat[4]
            #if playerStat[5] < 0:
            #    reward -= 30###############################################################################
            def ifdead(hero_sprites):
                dd = 0
                for hero in hero_sprites:
                    if dd == 0:
                        dd += 1
                        if hero.dead_time > 0:
                            return True
                return False

            reward = next_score - now_score
            if now_power - next_power > 1:#have power and touch bomb
                reward -= 25
            elif ifdead(hero_sprites):#no power and touch ghost or bomb
                reward -= 100
            #elif playerStat[3] = 0 and abs(next_x - now_x) + abs(next_y - now_y) > 5:#no power and touch ghost
            #    reward -= 100
            
            # Store sequence in replay memory
            agent.remember(state, move, reward, next_state, done)
            
            now_x = next_x
            now_y = next_y
            now_power = next_power
            now_score = next_score
            state = next_state
            game_score += reward
            reward -= 1  # Punish behavior which does not accumulate reward
            total_reward += reward
            
            if done:
                
                #print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                #    .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
                print("true score: {}, game score: {}, reward: {}, time:{}, total time: {}"
                    .format(now_score, game_score, total_reward, time, total_time))

                #if update:
                #    agent.update_target_model()
                
                agent.save('models/pacman_dqn')############################################################################### save
                pg.quit()
                return total_time
                #break
                
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        else:
            state = draw_action(mapmap, playerStat, otherPlayerStat, ghostStat, propsStat)
            now_x = playerStat[0]
            now_y = playerStat[1]
            now_power = playerStat[3]
            now_score = playerStat[4]
            #total_time = playerStat[6]   # Counter for total number of steps taken

        total_time += 1
        time += 1
        
        # Every update_rate timesteps we update the target network parameters
        if total_time % agent.update_rate == 0:
            agent.update_target_model()
            update = False

        step_two = True

        move = agent.act(state)
        player_action[0] = [move, random.choice([True, False])]
        
###############################################################################
        # receive control
        direction = [[-1, 0],[1, 0],[0, -1],[0, 1]]
        playerid = 0
        for hero in hero_sprites:
            if player_action[playerid][0] != 4:
                hero.changedirection(direction[player_action[playerid][0]], wall_sprites)
            hero.is_move = True
            if hero.landmine > 0 and player_action[playerid][1] and not (151 <= hero.rect.left <= 201 and 151 <= hero.rect.top <= 201):
                hero.superman_time = 10
                bomb_sprites.add(Food(hero.rect.left+8, hero.rect.top+8, 11, 11, BOMB_COLOR, BLACK))
                hero.landmine -= 1
            playerid += 1
        screen.fill(BLACK)  # black

        # update position and condition check
        for ghost in ghost_sprites:
            ghost.update(wall_sprites)
        for hero in hero_sprites:
            hero.update(wall_sprites)
            eat_pellet = pg.sprite.spritecollide(hero, pellet_sprites, True)
            eat_landmine = pg.sprite.spritecollide(hero, landmine_sprites, True)
            eat_power = pg.sprite.spritecollide(hero, power_sprites, True)
            if eat_pellet:
                level.Pellet_num -= len(eat_pellet)
                hero.score += 10
            if eat_landmine:
                level.Landmines_num -= len(eat_landmine)
                hero.landmine += 1
            if eat_power:
                hero.clock = pg.time.Clock()
                hero.super = True
                hero.super_time = 10000
                hero.speed = [speed * 8 / 5 for speed in hero.speed]
                hero.base_speed = [8, 8]
            if hero.super:
                dead_list = pg.sprite.spritecollide(hero, ghost_sprites, False)
                hero.clock.tick()
                hero.super_time -= hero.clock.get_time()
                if hero.super_time < 0:
                    hero.super = False
                    hero.speed = [speed * 5 / 8 for speed in hero.speed]
                    hero.base_speed = [5, 5]
                if dead_list:
                    for ghost in dead_list:
                        if (151 <= hero.rect.left <= 201 and 151 <= hero.rect.top <= 201): continue
                        # ghost eaten by hero
                        hero.score += 200
                        ghost_sprites.remove(ghost)
                        x=random.choice([176,201])
                        y=random.choice([176,201])
                        new_ghost = Ghost(x, y, GHOST + "blueGhost.png")
                        new_ghost.dead_time = 100
                        ghost_sprites.add(new_ghost)
        
        # condition check ( bomb )
        for bomb in bomb_sprites:
            dead_list = pg.sprite.spritecollide(bomb, hero_sprites, False)
            for hero in dead_list:
                if hero.superman_time == 0:
                    # hero dead by bomb
                    hero.dead_time = 100
                    hero.movePosition()
                    bomb_sprites.remove(bomb)
            dead_list = pg.sprite.spritecollide(bomb, ghost_sprites, False)
            if dead_list:
                bomb_sprites.remove(bomb)
                for ghost in dead_list:
                    # ghost dead by bomb
                    ghost_sprites.remove(ghost)
                    x = random.choice([176, 201])
                    y = random.choice([176, 201])
                    new_ghost = Ghost(x, y, GHOST + "blueGhost.png")
                    new_ghost.dead_time = 100
                    ghost_sprites.add(new_ghost)
        
        # condition check ( ghost )
        for ghost in ghost_sprites:
            if 151 <= ghost.rect.left <= 201 and 151 <= ghost.rect.top <= 201 : continue
            dead_list = pg.sprite.spritecollide(ghost, hero_sprites, False)
            for hero in dead_list:
                if 151 <= hero.rect.left <= 201 and 151 <= hero.rect.top <= 201:continue
                if hero.superman_time == 0:
                    # hero eaten by ghost
                    hero.dead_time = 100
                    hero.movePosition()
                            
        safe_place.draw(screen)
        hero_sprites.draw(screen)
        ghost_sprites.draw(screen)
        wall_sprites.draw(screen)
        pellet_sprites.draw(screen)
        landmine_sprites.draw(screen)
        power_sprites.draw(screen)
        bomb_sprites.draw(screen)
        
        color = ["yellow", "pink", "orange", "purple"]
        idx = 0
        text_height = 10
        for hero in hero_sprites:
            text_to_screen(screen, '{:6}: {}'.format(color[idx], hero.score), x=410, y=text_height)
            idx += 1
            text_height += 20
        text_to_screen(screen, 'time: {}'.format(STcpServer.idPackage), x=410, y=text_height)

        pg.display.flip()
        clock.tick(20)

    for i in range(4):
        status = STcpServer.Sendend(i)
    for hero in hero_sprites:
        print("{} : {}".format(hero.role_name, hero.score))


if __name__ == "__main__":
    tt = 0
    episodes = 1
    print("total time0: {}"
        .format(tt))
    for e in range(episodes):
        print("episode: {}/{}"
            .format(e+1, episodes))
        tt = main(tt)

    #os.system('pause')