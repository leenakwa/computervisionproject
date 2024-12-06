import pygame
from constants import FONT_SIZE, SMALL_FONT_SIZE


class Storage:
    """
    Singleton class for managing and storing game resources such as sounds, fonts, and images.
    """
    _instance = None

    def __new__(cls):
        # Ensures a single instance of the class is created
        if cls._instance is None:
            cls._instance = super(Storage, cls).__new__(cls)
            cls._instance._initialize_resources()
        return cls._instance

    def _initialize_resources(self):
        """
        Loads game resources (sounds, fonts, and images) into memory.
        """
        self.death = pygame.mixer.Sound('assets/sounds/sound_sfx_die.wav')
        self.swoosh = pygame.mixer.Sound('assets/sounds/sound_sfx_swooshing.wav')
        self.hit = pygame.mixer.Sound('assets/sounds/sound_sfx_hit.wav')
        self.font = pygame.font.Font('assets/fonts/04B_19.TTF', FONT_SIZE)
        self.font20 = pygame.font.Font('assets/fonts/04B_19.TTF', SMALL_FONT_SIZE)
        self.background = pygame.transform.scale2x(
            pygame.image.load('assets/images/background-day.png').convert())
        self.floor = pygame.transform.scale2x(
            pygame.image.load('assets/images/base.png').convert())
        self.bird_down_flap = pygame.transform.scale2x(
            pygame.image.load('assets/images/bluebird-downflap.png').convert_alpha())
        self.bird_mid_flap = pygame.transform.scale2x(
            pygame.image.load('assets/images/bluebird-midflap.png').convert_alpha())
        self.bird_up_flap = pygame.transform.scale2x(
            pygame.image.load('assets/images/bluebird-upflap.png').convert_alpha())
        self.pipe = pygame.transform.scale2x(
            pygame.image.load('assets/images/pipe-green.png').convert_alpha())
