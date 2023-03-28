import sys
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
from operator import add
import numpy as np

import pygame
import random
import time

# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

class SnakeGameEnv:
    def __init__(self):
        self.envName = "SnakeGameEnvironment"
        self.snakeSpeed = 15
        self.gameWidth = 320
        self.gameHeight = 320
        self.crashWindowBoundaryThreshold = 0
        self.oneStepMovement = 10
        self.action_space = np.arange(3) # 0 - move forward, 1- turn right, 2 - turn left
        self.reset()

    def reset(self):
        # Initialising pygame
        pygame.init()
        # Initialise game window
        pygame.display.set_caption('Snake Game - RL Agent')
        self.game_window = pygame.display.set_mode((self.gameWidth, self.gameHeight))
        # FPS (frames per second) controller
        self.fps = pygame.time.Clock()
        # defining snake default position
        self.snake_position = [100, 50]
        # defining first 4 blocks of snake body
        self.snake_body = [[100, 50],
                           [90, 50],
                           [80, 50],
                           [70, 50]
                           ]
        # fruit position
        self.fruit_position = [random.randrange(1, (self.gameWidth // 10)) * 10,
                               random.randrange(1, (self.gameHeight // 10)) * 10]
        self.fruit_spawn = True
        self.x_change = self.oneStepMovement
        self.y_change = 0
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.crash = False
        self.eaten = False
        self.done = False
        self.reward = 0
        self.no_of_steps = 0
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space
        self.no_of_steps += 1
        #action = 0 # 0 - move forward, 1- turn right, 2 - turn left
        if(self.direction=='RIGHT'):
            if action==0:
                self.x_change = self.oneStepMovement
                self.y_change = 0
            elif action==1:
                self.x_change = 0
                self.y_change = self.oneStepMovement
                self.direction = 'DOWN'
            elif action==2:
                self.x_change = 0
                self.y_change = -self.oneStepMovement
                self.direction = 'UP'
        elif(self.direction=='LEFT'):
            if action==0:
                self.x_change = -self.oneStepMovement
                self.y_change = 0
            elif action==1:
                self.x_change = 0
                self.y_change = -self.oneStepMovement
                self.direction = 'UP'
            elif action==2:
                self.x_change=0
                self.y_change=self.oneStepMovement
                self.direction = 'DOWN'
        elif(self.direction=='UP'):
            if action==0:
                self.x_change = 0
                self.y_change = -self.oneStepMovement
            elif action==1:
                self.x_change = self.oneStepMovement
                self.y_change = 0
                self.direction = 'RIGHT'
            elif action==2:
                self.x_change = -self.oneStepMovement
                self.y_change = 0
                self.direction = 'LEFT'
        elif(self.direction=='DOWN'):
            if action==0:
                self.x_change = 0
                self.y_change = self.oneStepMovement
            elif action==1:
                self.x_change = -self.oneStepMovement
                self.y_change = 0
                self.direction= 'LEFT'
            elif action==2:
                self.x_change = self.oneStepMovement
                self.y_change = 0
                self.direction = 'RIGHT'
        self.updatePosition()
        self.calculateReward()
        return self._get_obs(), self.reward, self.done, 0

    def calculateReward(self):
        if self.crash:
            self.reward = -10
        else:
            if self.eaten:
                self.reward = 10
                self.eaten = False
            else:
                self.reward = 0

    def showScore(self, choice, color, font, size):
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)

        # create the display surface object
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, color)

        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        # displaying text
        self.game_window.blit(score_surface, score_rect)

    def updatePosition(self):
        self.snake_position[0] = self.snake_position[0] + self.x_change
        self.snake_position[1] = self.snake_position[1] + self.y_change
        self.crashCheck()
        if (self.crash == False):
            self.foodEatenCheck()
            self.snake_body.insert(0, list(self.snake_position))
            if self.eaten:
                self.score += 10
                self.fruit_position = [random.randrange(1, (self.gameWidth // 10)) * 10,
                                       random.randrange(1, (self.gameHeight // 10)) * 10]
            else:
                self.snake_body.pop()
            self.game_window.fill(black)
            for pos in self.snake_body:
                pygame.draw.rect(self.game_window, green,
                                 pygame.Rect(pos[0], pos[1], self.oneStepMovement, self.oneStepMovement))
            pygame.draw.rect(self.game_window, white, pygame.Rect(
                self.fruit_position[0], self.fruit_position[1], self.oneStepMovement, self.oneStepMovement))
            # displaying score countinuously
            self.showScore(1, white, 'times new roman', 10)
            # Refresh game screen
            pygame.display.update()
            # Frame Per Second /Refresh Rate
            self.fps.tick(self.snakeSpeed)
        else:
            # deactivating pygame library
            # creating font object my_font
            my_font = pygame.font.SysFont('times new roman', 20)

            # creating a text surface on which text
            # will be drawn
            game_over_surface = my_font.render(
                'Your Score is : ' + str(self.score), True, red)

            # create a rectangular object for the text
            # surface object
            game_over_rect = game_over_surface.get_rect()

            # setting position of the text
            game_over_rect.midtop = (self.gameWidth / 2, self.gameHeight / 4)

            # blit will draw the text on screen
            self.game_window.blit(game_over_surface, game_over_rect)
            pygame.display.flip()

            # after 2 seconds we will quit the program
            time.sleep(2)
            pygame.quit()
            self.done = True

    def foodEatenCheck(self):
        if((self.fruit_position[0] == self.snake_position[0]) and (self.fruit_position[1] == self.snake_position[1])):
            self.eaten = True

    def crashCheck(self):

        if(self.snake_position in self.snake_body):
            self.crash = True
        elif((self.snake_position[0] < self.crashWindowBoundaryThreshold) or (self.snake_position[0]  >= (self.gameWidth - self.crashWindowBoundaryThreshold))):
            self.crash = True
        elif((self.snake_position[1] < self.crashWindowBoundaryThreshold) or (self.snake_position[1] >= (self.gameHeight - self.crashWindowBoundaryThreshold))):
            self.crash = True

    def _get_obs(self):
        state = [
        (self.x_change == self.oneStepMovement and self.y_change == 0 and (
                    (list(map(add, self.snake_body[0], [self.oneStepMovement, 0])) in self.snake_body) or
                    self.snake_body[0][0] + self.oneStepMovement >= (self.gameWidth - self.crashWindowBoundaryThreshold))) or \
        (self.x_change == -self.oneStepMovement and self.y_change == 0 and (
                        (list(map(add, self.snake_body[0], [-self.oneStepMovement, 0])) in self.snake_body) or
                        self.snake_body[0][0] - self.oneStepMovement <= self.crashWindowBoundaryThreshold)) or \
        (self.x_change == 0 and self.y_change == -self.oneStepMovement and (
                    (list(map(add, self.snake_body[0], [0, -self.oneStepMovement])) in self.snake_body) or
                    self.snake_body[0][1] - self.oneStepMovement <= self.crashWindowBoundaryThreshold)) or \
        (self.x_change == 0 and self.y_change == self.oneStepMovement and (
                    (list(map(add, self.snake_body[0], [0, self.oneStepMovement])) in self.snake_body) or
                    self.snake_body[0][1] + self.oneStepMovement >= (self.gameHeight - self.crashWindowBoundaryThreshold))), # danger straight

        (self.x_change == 0 and self.y_change == -self.oneStepMovement and (
                    (list(map(add, self.snake_body[0], [self.oneStepMovement, 0])) in self.snake_body) or
                    self.snake_body[0][0] + self.oneStepMovement >= (self.gameWidth - self.crashWindowBoundaryThreshold))) or
        (self.x_change == 0 and self.y_change == self.oneStepMovement and ((list(map(add, self.snake_body[0],[-self.oneStepMovement, 0])) in self.snake_body) or
                    self.snake_body[0][0] - self.oneStepMovement <= self.crashWindowBoundaryThreshold)) or (
                    self.x_change == -self.oneStepMovement and self.y_change == 0 and ((list(map(add, self.snake_body[0], [0, -self.oneStepMovement])) in self.snake_body) or self.snake_body[0][1] - self.oneStepMovement <= self.crashWindowBoundaryThreshold)) or
        (self.x_change == self.oneStepMovement and self.y_change == 0 and (
                    (list(map(add, self.snake_body[0], [0, self.oneStepMovement])) in self.snake_body) or self.snake_body[0][1] + self.oneStepMovement >= (self.gameHeight - self.crashWindowBoundaryThreshold))),  # danger right

        (self.x_change == 0 and self.y_change == self.oneStepMovement and (
                    (list(map(add, self.snake_body[0], [self.oneStepMovement, 0])) in self.snake_body) or
                    self.snake_body[0][0] + self.oneStepMovement >= (self.gameWidth - self.crashWindowBoundaryThreshold))) or
        (self.x_change == 0 and self.y_change == -self.oneStepMovement and ((list(map(
                add, self.snake_body[0], [-self.oneStepMovement, 0])) in self.snake_body) or self.snake_body[0][0] - self.oneStepMovement <= self.crashWindowBoundaryThreshold)) or (
                    self.x_change == self.oneStepMovement and self.y_change == 0 and (
                    (list(map(add, self.snake_body[0], [0, -self.oneStepMovement])) in self.snake_body) or self.snake_body[0][1] - self.oneStepMovement <= self.crashWindowBoundaryThreshold)) or (
                self.x_change == -self.oneStepMovement and self.y_change == 0 and (
                    (list(map(add, self.snake_body[0], [0, self.oneStepMovement])) in self.snake_body) or
                    self.snake_body[0][1] + self.oneStepMovement >= (self.gameHeight - self.crashWindowBoundaryThreshold))),  # danger left

            self.x_change == -self.oneStepMovement,  # move left
            self.x_change == self.oneStepMovement,  # move right
            self.y_change == -self.oneStepMovement,  # move up
            self.y_change == self.oneStepMovement,  # move down
            self.fruit_position[0] < self.snake_body[0][0],  # food left
            self.fruit_position[0] > self.snake_body[0][0],  # food right
            self.fruit_position[1] < self.snake_body[0][1],  # food up
            self.fruit_position[1] > self.snake_body[0][1]  # food down
            ]
        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0
        return np.array(state) #.reshape(1, -1)