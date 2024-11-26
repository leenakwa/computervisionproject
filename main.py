import pygame
import random
import sys


class EventFacade:
    def __init__(self):
        self.events = []
        self.keys_pressed = set()

    def handle_events(self) -> None:
        self.events = pygame.event.get()
        self.keys_pressed.clear()
        for event in self.events:
            if event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)

    def is_quit(self) -> bool:
        return any(event.type == pygame.QUIT for event in self.events)

    def is_key_pressed(self, key) -> bool:
        return key in self.keys_pressed

    def is_event_type(self, event_type) -> bool:
        return any(event.type == event_type for event in self.events)


class Storage:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Storage, cls).__new__(cls)
            cls._instance._initialize_resources()
        return cls._instance

    def _initialize_resources(self):
        self.death = pygame.mixer.Sound('sound_sfx_die.wav')
        self.swoosh = pygame.mixer.Sound('sound_sfx_swooshing.wav')
        self.hit = pygame.mixer.Sound('sound_sfx_hit.wav')
        self.font = pygame.font.Font('04B_19.TTF', FONT_SIZE)
        self.background = pygame.transform.scale2x(pygame.image.load('background-day.png').convert())
        self.floor = pygame.transform.scale2x(pygame.image.load('base.png').convert())
        self.bird_down_flap = pygame.transform.scale2x(pygame.image.load('bluebird-downflap.png').convert_alpha())
        self.bird_mid_flap = pygame.transform.scale2x(pygame.image.load('bluebird-midflap.png').convert_alpha())
        self.bird_up_flap = pygame.transform.scale2x(pygame.image.load('bluebird-upflap.png').convert_alpha())
        self.pipe = pygame.transform.scale2x(pygame.image.load('pipe-green.png').convert_alpha())


class Pipe:
    def __init__(self):
        self.pipe_img = storage.pipe
        self.screen = screen
        self.pipes = list()
        self.height = PIPE_HEIGHT

    def create_pipe(self):
        pipe_pos = random.choice(self.height)
        new_pipe = self.pipe_img.get_rect(midtop=(round(SCREEN_WIDTH), pipe_pos))
        top_pipe = self.pipe_img.get_rect(midbottom=(round(SCREEN_WIDTH), pipe_pos - PIPE_SPACING))
        self.pipes.append([new_pipe, top_pipe])

    def move_pipes(self):
        self.pipes = [pipe for pipe in self.pipes if pipe[0].right > 0]
        for pair in self.pipes:
            pair[0].centerx -= PIPE_SPEED
            pair[1].centerx -= PIPE_SPEED

    def check_collision(self, bird_rect):
        for pair in self.pipes:
            if bird_rect.colliderect(pair[0]) or bird_rect.colliderect(pair[1]):
                storage.death.play()
                return False
        if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT - 100:  # Учет пола
            storage.death.play()
            return False
        return True

    def draw_pipes(self):
        for pair in self.pipes:
            self.screen.blit(self.pipe_img, pair[0])
            flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
            self.screen.blit(flipped_pipe, pair[1])



class Bird:
    def __init__(self):
        self.frames = [storage.bird_up_flap, storage.bird_down_flap, storage.bird_mid_flap]
        self.index = 0
        self.image = self.frames[self.index]
        self.rect = self.image.get_rect(center=(100, 512))
        self.movement = 0
        self.screen = screen

    def rotate(self):
        rotated_bird = pygame.transform.rotozoom(self.image, -self.movement * 3, 1)
        return rotated_bird

    def animate(self):
        self.index = (self.index + 1) % len(self.frames)
        self.image = self.frames[self.index]
        self.rect = self.image.get_rect(center=(100, self.rect.centery))

    def update_movement(self, gravity):
        self.movement += gravity
        self.rect.centery += self.movement

    def draw(self):
        rotated_bird = self.rotate()
        self.screen.blit(rotated_bird, self.rect)

    def restart(self):
        self.rect.center = (100, SCREEN_HEIGHT // 2)
        self.movement = 0
        self.index = 0
        self.image = self.frames[self.index]


def draw_floor():
    screen.blit(floor_surface, (floor_x_pos, SCREEN_HEIGHT - 100))
    screen.blit(floor_surface, (floor_x_pos + SCREEN_WIDTH, SCREEN_HEIGHT - 100))


def score_display(game_active):
    score_surface = storage.font.render(f'Score: {int(score)}', True, (255, 255, 255))
    score_rect = score_surface.get_rect(center=(SCREEN_WIDTH / 2, 50))
    screen.blit(score_surface, score_rect)

    if game_active is False:
        high_score_surface = storage.font.render(f'High Score: {int(high_score)}', True, (255, 255, 255))
        high_score_rect = high_score_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        screen.blit(high_score_surface, high_score_rect)


def update_score(score, high_score):
    return max(score, high_score)


PIPE_HEIGHT = [400, 600, 800]
SCREEN_WIDTH, SCREEN_HEIGHT = 576, 1024
PIPE_SPACING = 300
PIPE_SPEED = 5
FONT_SIZE = 40
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
storage = Storage()
clock = pygame.time.Clock()
gravity = 0.25
bird_movement = 0
game_active = True
score = 0
floor_x_pos = 0
high_score = 0
background_surface = storage.background
floor_surface = storage.floor
BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP, 200)
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 1200)


# Создание объектов классов
event_facade = EventFacade()
bird = Bird()
pipe_manager = Pipe()

while True:
    event_facade.handle_events()
    if event_facade.is_quit():
        pygame.quit()
        sys.exit()
    if event_facade.is_key_pressed(pygame.K_SPACE):
        if game_active:
            bird.movement = -12
        else:
            game_active = True
            pipe_manager.pipes = []
            bird.restart()
            score = 0
    if event_facade.is_event_type(SPAWNPIPE):
        pipe_manager.create_pipe()
    if event_facade.is_event_type(BIRDFLAP):
        bird.animate()
    screen.blit(background_surface, (0, 0))
    if game_active:
        bird.update_movement(gravity)
        bird.draw()
        pipe_manager.move_pipes()
        pipe_manager.draw_pipes()
        game_active = pipe_manager.check_collision(bird.rect)
        score += 0.01
        score_display(game_active)
    else:
        high_score = update_score(score, high_score)
        score_display(game_active)
    floor_x_pos -= 1
    draw_floor()
    if floor_x_pos <= -SCREEN_WIDTH:
        floor_x_pos = 0
    pygame.display.update()
    clock.tick(120)
