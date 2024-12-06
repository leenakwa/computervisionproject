import random
import sys

import cv2
import mediapipe as mp
import numpy as np
import pygame


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
        self.death = pygame.mixer.Sound('assets/sounds/sound_sfx_die.wav')
        self.swoosh = pygame.mixer.Sound('assets/sounds/sound_sfx_swooshing.wav')
        self.hit = pygame.mixer.Sound('assets/sounds/sound_sfx_hit.wav')
        self.font = pygame.font.Font('assets/fonts/04B_19.TTF', FONT_SIZE)
        self.font20 = pygame.font.Font('assets/fonts/04B_19.TTF', SMALL_FONT_SIZE)
        self.background = pygame.transform.scale2x(pygame.image.load('assets/images/background-day.png').convert())
        self.floor = pygame.transform.scale2x(pygame.image.load('assets/images/base.png').convert())
        self.bird_down_flap = pygame.transform.scale2x(
            pygame.image.load('assets/images/bluebird-downflap.png').convert_alpha())
        self.bird_mid_flap = pygame.transform.scale2x(
            pygame.image.load('assets/images/bluebird-midflap.png').convert_alpha())
        self.bird_up_flap = pygame.transform.scale2x(
            pygame.image.load('assets/images/bluebird-upflap.png').convert_alpha())
        self.pipe = pygame.transform.scale2x(pygame.image.load('assets/images/pipe-green.png').convert_alpha())


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
        self.pipes.append({'bottom': new_pipe, 'top': top_pipe, 'passed': False})

    def move_pipes(self):
        self.pipes = [pipe for pipe in self.pipes if pipe['bottom'].right > 0]
        for pipe in self.pipes:
            pipe['bottom'].centerx -= PIPE_SPEED
            pipe['top'].centerx -= PIPE_SPEED

    def check_collision(self, bird_rect):
        for pipe in self.pipes:
            if bird_rect.colliderect(pipe['bottom']) or bird_rect.colliderect(pipe['top']):
                storage.death.play()
                return False
        if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT - storage.floor.get_size()[1] // 2:  # Учет пола
            storage.death.play()
            return False
        return True

    def draw_pipes(self):
        for pipe in self.pipes:
            self.screen.blit(self.pipe_img, pipe['bottom'])
            flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
            self.screen.blit(flipped_pipe, pipe['top'])

    def passing_counter(self):
        count = 0
        for pipe in pipe_manager.pipes:
            if not pipe['passed'] and bird.rect.centerx > pipe['bottom'].centerx:
                pipe['passed'] = True
                count += 1
        return count

    def restart(self):
        self.pipes = []


class Bird:
    def __init__(self):
        self.frames = [storage.bird_up_flap, storage.bird_down_flap, storage.bird_mid_flap]
        self.index = 0
        self.image = self.frames[self.index]
        self.rect = self.image.get_rect(center=(BIRD_START_COORD[0], BIRD_START_COORD[1]))
        self.movement = 0
        self.screen = screen
        self.visible = True

    def rotate(self):
        rotated_bird = pygame.transform.rotozoom(self.image, -self.movement * ROTATION_SPEED_MULTIPLIER, 1)
        return rotated_bird

    def animate(self):
        self.index = (self.index + 1) % len(self.frames)
        self.image = self.frames[self.index]
        self.rect = self.image.get_rect(center=(MARGIN, self.rect.centery))

    def update_movement(self, gravity):
        self.movement += gravity
        self.rect.centery += self.movement

    def draw(self):
        rotated_bird = self.rotate()
        self.screen.blit(rotated_bird, self.rect)

    def restart(self):
        self.rect.center = (MARGIN, SCREEN_HEIGHT // 2)
        self.movement = 0
        self.index = 0
        self.image = self.frames[self.index]
        self.visible = True


class EARFilter:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def update(self, ear_value):
        self.buffer.append(ear_value)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return np.mean(self.buffer)


def draw_floor():
    screen.blit(storage.floor, (floor_x_pos, SCREEN_HEIGHT - storage.floor.get_size()[1] // 2))
    screen.blit(storage.floor, (floor_x_pos + SCREEN_WIDTH, SCREEN_HEIGHT - storage.floor.get_size()[1] // 2))


def score_display(game_active):
    score_surface = storage.font.render(f'Score: {int(score)}', True, WHITE)
    score_rect = score_surface.get_rect(center=(SCREEN_WIDTH / 2, 50))
    screen.blit(score_surface, score_rect)

    if game_active is False:
        best_score_surface = storage.font.render(f'Best Score: {int(best_score)}', True, WHITE)
        best_score_rect = best_score_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        screen.blit(best_score_surface, best_score_rect)


def update_score(score, best):
    return max(score, best)


def calculate_ear(eye_points, landmarks):
    """Calculate Eye Aspect Ratio (EAR)"""
    d1 = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
    d2 = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
    d3 = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
    return (d1 + d2) / (2.0 * d3)


def display_blink_count(blink_count):
    blink_surface = storage.font.render(f'Blinks: {blink_count}', True, WHITE)
    blink_rect = blink_surface.get_rect(center=(SCREEN_WIDTH / 2, MARGIN))
    screen.blit(blink_surface, blink_rect)


def get_eye_regions(frame, landmarks):
    def extract_eye(eye_indices):
        coords = np.array([(int(landmarks[idx][0] * frame.shape[1]),
                            int(landmarks[idx][1] * frame.shape[0])) for idx in eye_indices])
        x, y, w, h = cv2.boundingRect(coords)
        return frame[y:y + h, x:x + w]

    left_eye = extract_eye(LEFT_EYE)
    right_eye = extract_eye(RIGHT_EYE)

    return left_eye, right_eye


def cv2_to_pygame(cv2_image):
    if cv2_image is None or cv2_image.size == 0:
        return None
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.flip(rgb_image, axis=1))


def enhance_eye_image(eye_image):
    if eye_image.size == 0:
        return None
    denoised_eye = cv2.bilateralFilter(eye_image, d=BILATERAL_FILTER_D, sigmaColor=BILATERAL_FILTER_SIGMA_COLOR,
                                       sigmaSpace=BILATERAL_FILTER_SIGMA_SPACE)
    sharp_eye = cv2.filter2D(denoised_eye, -1, FILTER_KERNEL)
    return sharp_eye


def eye_calculation(landmarks):
    left_ear = left_ear_filter.update(calculate_ear(LEFT_EYE, landmarks))
    right_ear = right_ear_filter.update(calculate_ear(RIGHT_EYE, landmarks))
    left_eye, right_eye = get_eye_regions(frame, landmarks)
    if left_eye is not None:
        left_eye = enhance_eye_image(left_eye)
        left_eye_surface = cv2_to_pygame(left_eye)
    if right_eye is not None:
        right_eye = enhance_eye_image(right_eye)
        right_eye_surface = cv2_to_pygame(right_eye)
    return [left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD, left_eye_surface, right_eye_surface]


def draw_eyes(left_eye_surface, right_eye_surface):
    if left_eye_surface:
        left_eye = pygame.transform.scale2x(
            pygame.transform.flip(pygame.transform.rotate(left_eye_surface, EYE_ROTATION_ANGLE), True, False))
        screen.blit(left_eye, EYE_COORD[0])

    if right_eye_surface:
        right_eye = pygame.transform.scale2x(
            pygame.transform.flip(pygame.transform.rotate(right_eye_surface, EYE_ROTATION_ANGLE), True, False))
        screen.blit(right_eye, EYE_COORD[1])


def write_text(text, font, color, coord):
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=coord)
    screen.blit(surface, rect)


def manage_floor(x):
    x -= 1
    draw_floor()
    if x <= -SCREEN_WIDTH:
        x = 0
    return x


pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 576, 1024
GRAVITY = 1
FPS = 60
SMALL_TEXT_COORD = (SCREEN_WIDTH // 2, 200)
EYE_COORD = [(SCREEN_WIDTH - 150, 50), (50, 50)]
BIRD_START_COORD = [100, 512]
ROTATION_SPEED_MULTIPLIER = 3
FLOOR_HEIGHT = 100
MARGIN = 100
SCORE_DISPLAY_Y = 50
PIPE_HEIGHT = [400, 600, 800]
PIPE_SPACING = 350
PIPE_SPEED = 15
FONT_SIZE = 40
SMALL_FONT_SIZE = 20
WHITE = (255, 255, 255)
RED = (190, 40, 40)
FLAP_STRENGTH = -22
EAR_THRESHOLD = 0.4
EAR_BUFFER_SIZE = 5
EAR_DENOMINATOR = 2.0
BILATERAL_FILTER_D = 7
BILATERAL_FILTER_SIGMA_COLOR = 60
BILATERAL_FILTER_SIGMA_SPACE = 75
FILTER_KERNEL = np.array([[0, -0.5, 0],
                          [-0.5, 3, -0.5],
                          [0, -0.5, 0]])
EYE_SCALE_FACTOR = 2
EYE_ROTATION_ANGLE = -90
CONSECUTIVE_FRAMES = 2
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
START = 'start'
GAME = 'game'
END = 'end'

instructions_text = ['Bird jumps when you blink.',  'Press SPACE to start']

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
storage = Storage()
cap = cv2.VideoCapture(0)
event_facade = EventFacade()
bird = Bird()
pipe_manager = Pipe()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
left_ear_filter = EARFilter(EAR_BUFFER_SIZE)
right_ear_filter = EARFilter(EAR_BUFFER_SIZE)

game_state = START
game_active = True
bird_movement = 0
floor_x_pos = 0
score = 0
best_score = 0
blink_count = 0
frame_counter = 0

BIRDFLAP = pygame.USEREVENT + 1
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(BIRDFLAP, 100)
pygame.time.set_timer(SPAWNPIPE, 1600)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    event_facade.handle_events()
    if event_facade.is_quit():
        pygame.quit()
        sys.exit()
    if event_facade.is_key_pressed(pygame.K_SPACE):
        if game_state == END or game_state == START:
            game_state = GAME
            pipe_manager.pipes = []
            bird.restart()
            score = 0
    if event_facade.is_event_type(SPAWNPIPE):
        pipe_manager.create_pipe()
    if event_facade.is_event_type(BIRDFLAP):
        bird.animate()

    screen.blit(storage.background, (0, 0))
    if game_state == START:
        write_text(instructions_text[0], storage.font20, WHITE, SMALL_TEXT_COORD)
        write_text(instructions_text[1], storage.font20, RED, (SMALL_TEXT_COORD[0], SMALL_TEXT_COORD[1] + MARGIN))
    if game_state == GAME:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        left_eye_surface, right_eye_surface = None, None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                eye_bool, left_eye_surface, right_eye_surface = eye_calculation(landmarks)
                if eye_bool:
                    frame_counter += 1
                else:
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        blink_count += 1
                        bird.movement = FLAP_STRENGTH
                        storage.swoosh.play()
                    frame_counter = 0
        draw_eyes(left_eye_surface, right_eye_surface)
        bird.update_movement(GRAVITY)
        bird.draw()
        pipe_manager.move_pipes()
        pipe_manager.draw_pipes()
        score += pipe_manager.passing_counter()
        game_active = pipe_manager.check_collision(bird.rect)
        score_display(game_active)
        if not game_active:
            game_state = END
    elif game_state == END:
        best_score = update_score(score, best_score)
        score_display(False)
        storage.death.play()
        bird.restart()
        pipe_manager.restart()
        score = 0
        frame_counter = 0
        game_state = START

    floor_x_pos = manage_floor(floor_x_pos)
    display_blink_count(blink_count)
    score_display(game_active)
    pygame.display.update()
    clock.tick(FPS)
