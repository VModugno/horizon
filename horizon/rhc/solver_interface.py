from dataclasses import dataclass, field

@dataclass
class SolverInterface:
    type: str = 'ipopt'
    opts: dict = field(default_factory=dict)
    receding: bool = False

