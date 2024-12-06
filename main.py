import random
import sys

import cv2
import mediapipe as mp
import pygame

from constants import *
from event_facade import EventFacade
from storage import Storage


class Pipe:
    """
    Manages pipes in the game, including their creation, movement, drawing, collision detection,
    and scoring logic.
    """

    def __init__(self):
        self.pipe_img = storage.pipe  # Pipe image from storage
        self.screen = screen  # Game screen
        self.pipes = list()  # List of pipe dictionaries
        self.height = PIPE_HEIGHT  # Possible heights for pipes

    def create_pipe(self):
        """
        Creates a new set of pipes (top and bottom) with random height.
        """
        pipe_pos = random.choice(self.height)
        new_pipe = self.pipe_img.get_rect(midtop=(round(SCREEN_WIDTH), pipe_pos))
        top_pipe = self.pipe_img.get_rect(midbottom=(round(SCREEN_WIDTH), pipe_pos - PIPE_SPACING))
        self.pipes.append({'bottom': new_pipe, 'top': top_pipe, 'passed': False})

    def move_pipes(self):
        """
        Moves pipes to the left and removes off-screen pipes.
        """
        self.pipes = [pipe for pipe in self.pipes if pipe['bottom'].right > 0]
        for pipe in self.pipes:
            pipe['bottom'].centerx -= PIPE_SPEED
            pipe['top'].centerx -= PIPE_SPEED

    def check_collision(self, bird_rect):
        """
        Checks for collisions between the bird and pipes, or the screen boundaries.
        Plays death sound and ends the game if a collision occurs.
        """
        for pipe in self.pipes:
            if bird_rect.colliderect(pipe['bottom']) or bird_rect.colliderect(pipe['top']):
                storage.death.play()
                return False
        # Check collision with the top of the screen or the floor
        if bird_rect.top <= 0 or bird_rect.bottom >= SCREEN_HEIGHT - storage.floor.get_size()[1] // 2:
            storage.death.play()
            return False
        return True

    def draw_pipes(self):
        """
        Draws all pipes on the screen, including flipped top pipes.
        """
        for pipe in self.pipes:
            self.screen.blit(self.pipe_img, pipe['bottom'])
            flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
            self.screen.blit(flipped_pipe, pipe['top'])

    def passing_counter(self):
        """
        Counts and returns the number of pipes passed by the bird.
        """
        count = 0
        for pipe in self.pipes:
            if not pipe['passed'] and bird.rect.centerx > pipe['bottom'].centerx:
                pipe['passed'] = True
                count += 1
        return count

    def restart(self):
        """
        Resets the pipes for a new game.
        """
        self.pipes = []


class Bird:
    """
    Represents the bird character in the game, including its movement, animation,
    rotation, and reset functionality.
    """

    def __init__(self):
        self.frames = [storage.bird_up_flap, storage.bird_down_flap, storage.bird_mid_flap]  # Animation frames
        self.index = 0  # Current animation frame index
        self.image = self.frames[self.index]  # Current image of the bird
        self.rect = self.image.get_rect(center=(BIRD_START_COORD[0], BIRD_START_COORD[1]))  # Bird's position
        self.movement = 0  # Vertical movement speed
        self.screen = screen  # Game screen
        self.visible = True  # Visibility status for the bird

    def rotate(self):
        """
        Rotates the bird image based on its movement speed for a realistic falling effect.
        """
        rotated_bird = pygame.transform.rotozoom(
            self.image, -self.movement * ROTATION_SPEED_MULTIPLIER, 1)
        return rotated_bird

    def animate(self):
        """
        Updates the bird's animation by cycling through its frames.
        """
        self.index = (self.index + 1) % len(self.frames)
        self.image = self.frames[self.index]
        self.rect = self.image.get_rect(center=(MARGIN, self.rect.centery))

    def update_movement(self, gravity):
        """
        Updates the bird's vertical movement based on gravity.
        """
        self.movement += gravity
        self.rect.centery += self.movement

    def draw(self):
        """
        Draws the bird on the screen with its current rotation.
        """
        rotated_bird = self.rotate()
        self.screen.blit(rotated_bird, self.rect)

    def restart(self):
        """
        Resets the bird's position, movement, and animation for a new game.
        """
        self.rect.center = (MARGIN, SCREEN_HEIGHT // 2)
        self.movement = 0
        self.index = 0
        self.image = self.frames[self.index]
        self.visible = True


class EARFilter:
    """
    A class for filtering and smoothing Eye Aspect Ratio (EAR) values using a sliding buffer.
    """

    def __init__(self, buffer_size):
        """
        Initializes the EARFilter with a specified buffer size.

        :param buffer_size: The maximum number of EAR values to retain in the buffer.
        """
        self.buffer = []  # List to store recent EAR values
        self.buffer_size = buffer_size  # Maximum size of the buffer

    def update(self, ear_value):
        """
        Updates the buffer with a new EAR value and calculates the smoothed average.

        :param ear_value: The latest EAR value to add to the buffer.
        :return: The mean of the values in the buffer.
        """
        self.buffer.append(ear_value)
        if len(self.buffer) > self.buffer_size:  # Ensure the buffer size does not exceed the limit
            self.buffer.pop(0)  # Remove the oldest value from the buffer
        return np.mean(self.buffer)  # Return the smoothed EAR value


class Eye:
    def __init__(self, left_eye_indices, right_eye_indices, ear_threshold, ear_buffer_size):
        """
        Initialize the Eye class with indices, EAR threshold, and buffer size.

        :param left_eye_indices: Indices of landmarks for the left eye.
        :param right_eye_indices: Indices of landmarks for the right eye.
        :param ear_threshold: Threshold to detect blinking.
        :param ear_buffer_size: Size of the buffer for EAR smoothing.
        """
        self.left_eye_indices = left_eye_indices
        self.right_eye_indices = right_eye_indices
        self.ear_threshold = ear_threshold
        self.left_ear_filter = EARFilter(ear_buffer_size)
        self.right_ear_filter = EARFilter(ear_buffer_size)

    @staticmethod
    def calculate_ear(eye_points, landmarks_eye):
        """Calculate Eye Aspect Ratio (EAR)."""
        d1 = np.linalg.norm(np.array(landmarks_eye[eye_points[1]]) - np.array(landmarks_eye[eye_points[5]]))
        d2 = np.linalg.norm(np.array(landmarks_eye[eye_points[2]]) - np.array(landmarks_eye[eye_points[4]]))
        d3 = np.linalg.norm(np.array(landmarks_eye[eye_points[0]]) - np.array(landmarks_eye[eye_points[3]]))
        return (d1 + d2) / (2.0 * d3)

    def update_ear(self, landmarks_eye):
        """Update EAR for both eyes and check if blinking."""
        left_ear = self.left_ear_filter.update(self.calculate_ear(self.left_eye_indices, landmarks_eye))
        right_ear = self.right_ear_filter.update(self.calculate_ear(self.right_eye_indices, landmarks_eye))
        return left_ear < self.ear_threshold and right_ear < self.ear_threshold

    @staticmethod
    def get_eye_regions(frame_eye, landmarks_eye, eye_indices):
        """Extract eye regions based on landmarks."""
        coordinates = np.array([(int(landmarks_eye[idx][0] * frame_eye.shape[1]),
                                 int(landmarks_eye[idx][1] * frame_eye.shape[0])) for idx in eye_indices])
        x, y, w, h = cv2.boundingRect(coordinates)
        return frame_eye[y:y + h, x:x + w]

    @staticmethod
    def enhance_eye_image(eye_image):
        """Enhance eye region image."""
        if eye_image.size == 0:
            return None
        denoised_eye = cv2.bilateralFilter(eye_image, d=BILATERAL_FILTER_D,
                                           sigmaColor=BILATERAL_FILTER_SIGMA_COLOR,
                                           sigmaSpace=BILATERAL_FILTER_SIGMA_SPACE)
        sharp_eye = cv2.filter2D(denoised_eye, -1, FILTER_KERNEL)
        return sharp_eye

    @staticmethod
    def cv2_to_pygame(cv2_image):
        """Convert an OpenCV image to a Pygame surface."""
        if cv2_image is None or cv2_image.size == 0:
            return None
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(np.flip(rgb_image, axis=1))

    def process_eyes(self, frame_eye, landmarks_eye):
        """
        Process eye regions and return EAR states and Pygame surfaces.

        :param frame_eye: Video frame to process.
        :param landmarks_eye: Facial landmarks.
        :return: Processed surfaces for left and right eyes.
        """
        left_eye = self.get_eye_regions(frame_eye, landmarks_eye, self.left_eye_indices)
        right_eye = self.get_eye_regions(frame_eye, landmarks_eye, self.right_eye_indices)

        left_eye_surf = self.cv2_to_pygame(self.enhance_eye_image(left_eye)) if left_eye is not None else None
        right_eye_surf = self.cv2_to_pygame(self.enhance_eye_image(right_eye)) if right_eye is not None else None

        return left_eye_surf, right_eye_surf

    @staticmethod
    def draw_eyes(py_screen, left_eye_surf, right_eye_surf, eye_coordinates, eye_rotation_angle):
        """
        Draw eyes on the screen.

        :param py_screen: Pygame screen to draw on.
        :param left_eye_surf: Pygame surface of the left eye.
        :param right_eye_surf: Pygame surface of the right eye.
        :param eye_coordinates: Coordinates for positioning the eyes.
        :param eye_rotation_angle: Angle for rotating the eye images.
        """
        if left_eye_surf:
            left_eye = pygame.transform.scale2x(
                pygame.transform.flip(pygame.transform.rotate(left_eye_surf, eye_rotation_angle), True, False))
            py_screen.blit(left_eye, eye_coordinates[0])

        if right_eye_surface:
            right_eye = pygame.transform.scale2x(
                pygame.transform.flip(pygame.transform.rotate(right_eye_surf, eye_rotation_angle), True, False))
            py_screen.blit(right_eye, eye_coordinates[1])


def draw_floor():
    """Draw the floor at the current and next position for seamless scrolling."""
    screen.blit(storage.floor, (floor_x_pos, SCREEN_HEIGHT - storage.floor.get_size()[1] // 2))
    screen.blit(storage.floor, (floor_x_pos + SCREEN_WIDTH, SCREEN_HEIGHT - storage.floor.get_size()[1] // 2))


def score_display(game_status):
    """Display the current score and, if inactive, show the best score."""
    score_surface = storage.font.render(f'Score: {int(score)}', True, WHITE)
    score_rect = score_surface.get_rect(center=(SCREEN_WIDTH / 2, 50))
    screen.blit(score_surface, score_rect)

    if not game_status:
        best_score_surface = storage.font.render(f'Best Score: {int(best_score)}', True, WHITE)
        best_score_rect = best_score_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        screen.blit(best_score_surface, best_score_rect)


def update_score(current_score, best):
    """Update the best score if the current score exceeds it."""
    return max(current_score, best)


def display_blink_count(blink_counter):
    """Show the number of blinks detected."""
    blink_surface = storage.font.render(f'Blinks: {blink_counter}', True, WHITE)
    blink_rect = blink_surface.get_rect(center=(SCREEN_WIDTH / 2, MARGIN))
    screen.blit(blink_surface, blink_rect)


def write_text(text, font, color, coord):
    """Render and draw text at the specified coordinates."""
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=coord)
    screen.blit(surface, rect)


def manage_floor(x):
    """Scroll the floor and return the updated x position."""
    x = (x - 1) % -SCREEN_WIDTH
    draw_floor()
    return x


# Set up the Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Initialize class objects
storage = Storage()
event_facade = EventFacade()
bird = Bird()
pipe_manager = Pipe()
eye_processor = Eye(LEFT_EYE, RIGHT_EYE, EAR_THRESHOLD, EAR_BUFFER_SIZE)
left_ear_filter = EARFilter(EAR_BUFFER_SIZE)
right_ear_filter = EARFilter(EAR_BUFFER_SIZE)

# Initialize OpenCV video capture for accessing camera
cap = cv2.VideoCapture(0)

# Set up MediaPipe FaceMesh for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Initialize game state variables
game_active = True
game_state = START
bird_movement = 0
floor_x_pos = 0
score = 0
best_score = 0
frame_counter = 0
blink_count = 0

# Set up user events for bird flapping and spawning pipes
BIRD_FLAP = pygame.USEREVENT + 1  # Custom event for bird flapping
SPAWN_PIPE = pygame.USEREVENT  # Custom event for spawning pipes

# Set the timer for the bird flapping event (every 100ms) and spawning pipes (every 1600ms)
pygame.time.set_timer(BIRD_FLAP, 100)
pygame.time.set_timer(SPAWN_PIPE, 1600)

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
    if event_facade.is_event_type(SPAWN_PIPE):
        pipe_manager.create_pipe()
    if event_facade.is_event_type(BIRD_FLAP):
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
                blink_detected = eye_processor.update_ear(landmarks)
                left_eye_surface, right_eye_surface = eye_processor.process_eyes(frame, landmarks)
                if blink_detected:
                    frame_counter += 1
                else:
                    if frame_counter >= CONSECUTIVE_FRAMES:
                        blink_count += 1
                        bird.movement = FLAP_STRENGTH
                        storage.swoosh.play()
                    frame_counter = 0
        eye_processor.draw_eyes(screen, left_eye_surface, right_eye_surface, EYE_COORD, EYE_ROTATION_ANGLE)
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
