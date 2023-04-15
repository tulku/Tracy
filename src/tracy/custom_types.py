import dataclasses


@dataclasses.dataclass
class ScreenConfig:
    width: int
    height: int
    scale: int

    def getScreenSize(self):
        return (self.width, self.height)

    def getTargetSize(self):
        return (self.width * self.scale, self.height * self.scale)
