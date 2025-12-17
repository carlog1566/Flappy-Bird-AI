# IMPORT LIBRARIES
import pygame
import random
from sys import exit

# DEFINE CONSTANTS
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 800
SPEED = 60
GROUND_HEIGHT = 100
GROUND_Y = WINDOW_HEIGHT - GROUND_HEIGHT
GRAVITY = 0.5

# DEFINE PLAYER ATTRIBUTES, POSITION, AND VELOCITY
player_width = 55
player_height = 40
player_x = 200
player_y = 400
player_y_velocity = 0
player_angle = 45

# DEFINE PIPE ATTRIBUTES & BEHAVIOR
pipe_width = 100
pipe_length = 400 
pipe_gap = 200
scroll_speed = 3
spawn_rate = 100
pipes = pygame.sprite.Group()

# INITIALIZE PYGAME
pygame.init()

# DEFINE SCORE AND SCORE FONT
score = 0
score_font = pygame.font.Font('resources/EXEPixelPerfect.ttf', 100)



class Pipe(pygame.sprite.Sprite):

    # PIPE CONSTRUCTOR
    def __init__(self, x, y, pos, display, scroll_speed):
        # Initialize sprite class and variables
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.scroll_speed = scroll_speed
        self.display = display

        
        # Create rect for top or bottom pipe
        if pos == 1:
            self.image = pygame.image.load('./resources/Pipe.png').convert_alpha()
            self.image = pygame.transform.scale(self.image, (pipe_width, pipe_length)) 
            self.rect = self.image.get_rect(bottomleft=(x, y))
            self.passed = False
        elif pos == -1:
            self.image = pygame.transform.flip(pygame.image.load('./resources/Pipe.png').convert_alpha(), False, True)
            self.image = pygame.transform.scale(self.image, (pipe_width, pipe_length))
            self.rect = self.image.get_rect(topleft=(x, y))  
            self.passed = True
        

    # UPDATE PIPE FUNCTION
    def update(self):
        # Update x
        self.rect.x -= self.scroll_speed

        # Update score if top pipe has passed the bird
        global score
        if (self.rect.right < player_x) and (not self.passed):
            self.passed = True
            score += 1
        
        # Erase pipe when the right side reaches x < -10
        if self.rect.right < -10:
            self.kill()
       


class Bird(pygame.sprite.Sprite):

    # BIRD CONSTRUCTOR
    def __init__(self, x, y, y_velocity, display):
        # Initialize sprite class and variables
        super().__init__()
        self.x = x
        self.y = y
        self.y_velocity = y_velocity
        self.angle = player_angle
        self.display = display

        self.image = pygame.transform.scale(pygame.image.load('./resources/FlappyBird.png').convert_alpha(), (player_width, player_height))
        self.image = pygame.transform.rotate(self.image, self.angle)
        self.rect = self.image.get_rect(midbottom=(self.x, self.y))


    # FLAP FUNCTION
    def flap(self):
        # Set y_velocity and angle
        self.y_velocity = -10
        self.angle = 45


    # UPDATE BIRD FUNCTION
    def update(self):
        # Simulate gravity and make bird more dynamic
        self.y_velocity += GRAVITY
        self.y += self.y_velocity
        if self.y_velocity > 0 and self.angle > -90:
            self.angle -= 3



class FlappyBird():

    # GAME CONSTRUCTOR
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
        top_pipe = Pipe(WINDOW_WIDTH, rand_y, 1, self.display, scroll_speed)
        bot_pipe = Pipe(WINDOW_WIDTH, rand_y + pipe_gap, -1, self.display, scroll_speed)

        # Add the pair into the sprite group
        group.add(top_pipe)
        group.add(bot_pipe)


    # COLLISION FUNCTION
    def is_collision(self):
        # Return True when player hits the ground or 100 units above the roof or collides with pipe else return False
        collided = pygame.sprite.spritecollide(self.player, pipes, False)
        if collided or (self.player.y > GROUND_Y) or (self.player.y < -100):
            return True
        return False


    # UPDATE FRAME FUNCTION
    def update(self):
        # Update display
        self.display.fill((173, 216, 230))

        # Update pipes & player if not game_over, else just update player
        if not self.game_over:
            self.player.update()
 
            if self.counter >= spawn_rate:
                self.spawn_pipe(pipes)
                self.counter = 0
            else:
                self.counter += 1

            pipes.update() 
        else:
            if self.player.y < GROUND_Y:
                self.player.update()
        
        # Update pipes
        for pipe in pipes:
            self.display.blit(pipe.image, pipe.rect) 

        # Update bird/player
        self.player.image = pygame.transform.scale(pygame.image.load('./resources/FlappyBird.png').convert_alpha(), (player_width, player_height))
        self.player.image = pygame.transform.rotate(self.player.image, self.player.angle)
        self.player.rect = self.player.image.get_rect(midbottom=(self.player.x, self.player.y))
        self.display.blit(self.player.image, self.player.rect)

        # Update score text
        self.score = score
        score_text = score_font.render(str(self.score), True, 'White')
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 80 ))
        self.display.blit(score_text, score_rect)

        # Display ground
        grass_rect = pygame.Rect((0, WINDOW_HEIGHT - GROUND_HEIGHT), (WINDOW_WIDTH, 20))
        ground_rect = pygame.Rect((0, WINDOW_HEIGHT - GROUND_HEIGHT), (WINDOW_WIDTH, GROUND_HEIGHT))
        pygame.draw.rect(self.display, 'antiquewhite2', ground_rect)
        pygame.draw.rect(self.display, 'chartreuse4', grass_rect)

        # Display all updates
        pygame.display.flip()


    # RESET GAME FUNCTION
    def reset(self):
        # Create player from Bird class and set spawn and game_over
        self.player = Bird(player_x, player_y, player_y_velocity, self.display)
        self.spawn = True
        self.game_over = False

        # Define counter for pipe spawn rate and empty sprite group
        self.counter = 0
        pipes.empty()

        # Define score in scope and set it at 0
        global score
        score = 0


    # PLAY FUNCTION
    def play(self):
        for event in pygame.event.get():
            # Quit game via exiting out of tab
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            # Update y velocity as spacebar is pressed
            if event.type == pygame.KEYDOWN and not self.game_over:
                if event.key == pygame.K_SPACE:
                    self.player.flap()
        
        # Check for collision
        if not self.game_over and self.is_collision():
            self.game_over = True

        # Update frame & set framerate
        self.update()
        self.clock.tick(SPEED)
            

# MAIN   
if __name__ == '__main__':
    # Create game object (FlappyBird)
    game = FlappyBird()

    while True:
        game.play()