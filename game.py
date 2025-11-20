import os
import pygame
import random
from sys import exit

# DEFINE CONSTANTS
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
SPEED = 60

GRAVITY = 0.5

# DEFINE PLAYER POSITION AND VELOCITY
player_x = 200
player_y = 400
player_y_velocity = 0

# INITIALIZE PYGAME
pygame.init()


class FlappyBird():

    # INITIALIZE GAME
    def __init__(self):
        self.w = WINDOW_WIDTH
        self.h = WINDOW_HEIGHT

        # Initialize Display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()
        
        # Call reset function to load inital state of game
        self.reset()


    # COLLISION FUNCTION
    def is_collision(self):
        pass


    # UPDATE FRAME FUNCTION
    def update(self):
        # Update display & player rect 
        self.display.fill('Black')
        player_rect = pygame.Rect((self.player_x, self.player_y), (20, 20))
        pygame.draw.rect(self.display, 'Red', player_rect)

        pygame.display.flip()


    # RESET GAME FUNCTION
    def reset(self):
        # Define player x, y, and y velocity
        self.player_x = player_x
        self.player_y = player_y
        self.player_y_velocity = player_y_velocity


    # PLAY FUNCTION
    def play(self):
        for event in pygame.event.get():
            # Quit game via exiting out of tab
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            # Update y velocity as spacebar is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.player_y_velocity = -10
        
        # Simulate flappy bird physics via y velocity and gravity
        self.player_y_velocity += GRAVITY
        self.player_y += self.player_y_velocity

        # Call update function to update frame and set framerate
        self.update()
        self.clock.tick(SPEED)
            

            

if __name__ == '__main__':
    # Create game object (FlappyBird)
    game = FlappyBird()

    # while True:
    #     game_over, score = game.play

    #     if game_over:
    #         break
    while True:
        game.play()


    pygame.quit()
    exit()
