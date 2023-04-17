import dataclasses


@dataclasses.dataclass
class ScreenConfig:
    width: int
    height: int
    scale: int

    def get_screen_size(self):
        return (self.width, self.height)

    def get_target_size(self):
        return (self.width * self.scale, self.height * self.scale)
