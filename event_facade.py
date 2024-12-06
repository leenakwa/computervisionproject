import pygame


class EventFacade:
    """
    A helper class for handling pygame events and key presses.
    """
    def __init__(self):
        self.events = []  # List of pygame events
        self.keys_pressed = set()  # Set of currently pressed keys

    def handle_events(self) -> None:
        """
        Updates the list of events and pressed keys.
        """
        self.events = pygame.event.get()
        self.keys_pressed.clear()
        for event in self.events:
            if event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)

    def is_quit(self) -> bool:
        """
        Checks if a quit event occurred.
        """
        return any(event.type == pygame.QUIT for event in self.events)

    def is_key_pressed(self, key) -> bool:
        """
        Checks if a specific key is pressed.
        """
        return key in self.keys_pressed

    def is_event_type(self, event_type) -> bool:
        """
        Checks if a specific event type exists.
        """
        return any(event.type == event_type for event in self.events)
