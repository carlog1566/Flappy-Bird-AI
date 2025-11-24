# IMPORT LIBRARIES
import os
import pygame
import random
from sys import exit

# DEFINE CONSTANTS
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
SPEED = 60
GROUND_Y = WINDOW_HEIGHT - 100
GROUND_HEIGHT = 100
GRAVITY = 0.5

# DEFINE PLAYER POSITION AND VELOCITY
player_x = 200
player_y = 400
player_y_velocity = 0

# DEFINE PIPE BEHAVIOR
pipe_gap = 150
scroll_speed = 3
spawn_rate = 100
pipes = pygame.sprite.Group()

# INITIALIZE PYGAME
pygame.init()


class Pipe(pygame.sprite.Sprite):

    # INITIALIZE PIPE OBJECT
    def __init__(self, x, y, pos, display, scroll_speed):
        # Initialize sprite class and variables
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.scroll_speed = scroll_speed
        self.display = display
        
        # Create rect for top or bottom pipe
        if pos == 1:
            self.rect = pygame.Rect((0, 0), (50, 500))
            self.rect.midbottom = (x, y)
        elif pos == -1:
            self.rect = pygame.Rect((0, 0), (50, 500))
            self.rect.midtop = (x, y)


    # UPDATE PIPE FUNCTION
    def update(self):
        # Display pipe and update x 
        pygame.draw.rect(self.display, 'Green', self.rect)
        self.rect.x -= self.scroll_speed
        
        # Erase pipe when the right side reaches x < -10
        if self.rect.right < -10:
            self.kill()


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

    
    # SPAWN PIPE FUNCTION
    def spawn_pipe(self, group):
        # Generate a random y coordinate for pipes
        rand_y = random.randint(GROUND_HEIGHT, WINDOW_HEIGHT - GROUND_HEIGHT - pipe_gap - 100)

        # Create a top and bottom pipe pair
        top_pipe = Pipe(WINDOW_WIDTH + 50, rand_y, 1, self.display, scroll_speed)
        bot_pipe = Pipe(WINDOW_WIDTH + 50, rand_y + pipe_gap, -1, self.display, scroll_speed)

        # Add the pair into the sprite group
        group.add(top_pipe)
        group.add(bot_pipe)


    # COLLISION FUNCTION
    def is_collision(self):
        # Return True when player hits the ground or 50 units above the roof else return False
        if self.player_y > GROUND_Y or self.player_y < -50:
            return True
        return False


    # UPDATE FRAME FUNCTION
    def update(self, spawn):
        # Update display & player rect 
        self.display.fill('Black')
        player_rect = pygame.Rect((self.player_x, self.player_y), (20, 20))
        pygame.draw.rect(self.display, 'Red', player_rect)

        # Spawn pipe when spawn is true
        if spawn:
            self.spawn_pipe(pipes)
            print('Pipe Spawned')

        # Update all pipes
        pipes.update()

        pygame.display.flip()


    # RESET GAME FUNCTION
    def reset(self):
        # Define player x, y, and y velocity
        self.player_x = player_x
        self.player_y = player_y
        self.player_y_velocity = player_y_velocity

        # Define counter for pipe spawn rate
        self.counter = 0


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

        # Set game_over equal to false; if collision game_over = true and game ends
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over

        # Update frame and spawn pipe if counter >= spawn_rate
        if self.counter >= spawn_rate:
            self.update(True)
            self.counter = 0
        else:
            self.update(False)
            self.counter += 1

        # Framerate
        self.clock.tick(SPEED)
            

            
if __name__ == '__main__':
    # Create game object (FlappyBird)
    game = FlappyBird()

    while True:
        game_over = game.play()

        if game_over:
            break

    # For Testing
    # while True:
    #     game.play()

    pygame.quit()
    exit()
