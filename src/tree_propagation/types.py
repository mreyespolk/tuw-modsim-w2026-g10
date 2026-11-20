from enum import StrEnum


class DispersalType(StrEnum):
    WIND = "wind"
    ANIMAL = "animal"
    GRAVITY = "gravity"

class AgentStatus(StrEnum):
    ALIVE = "Alive"
    DEAD = "Dead"