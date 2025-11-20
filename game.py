import os
import pygame
import random
from sys import exit

WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
SPEED = 10

pygame.init()

class FlappyBird():

    # INITIALIZE GAME
    def __init__(self):
        self.w = WINDOW_WIDTH
        self.h = WINDOW_HEIGHT

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()

        self.reset()

    # RESET GAME FUNCTION
    def reset(self):
        pass

    # COLLISION FUNCTION
    def is_collision(self):
        pass

    # UPDATE FRAME FUNCTION
    def update_frame(self):
        pass

    # PLAY FUNCTION
    def play(self):
        pass




if __name__ == '__main__':
    game = FlappyBird()

