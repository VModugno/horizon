from horizon.rhc.tasks.task import Task
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.rhc.tasks.limitsTask import JointLimitsTask, VelocityLimitsTask
from horizon.rhc.tasks.posturalTask import PosturalTask

from typing import Callable, Dict, Any

task_creation_funcs: Dict[str, Callable[..., Task]] = {}

def register(task_type: str, func: Callable[..., Task]) -> None:
    """
    Register a new task class.
    """
    task_creation_funcs[task_type] = func

def unregister(task_type: str) -> None:
    """
    Unregister a task class.
    """
    task_creation_funcs.pop(task_type, None)

def create(args: Dict[str, Any]) -> Task:
    """
    Create a new task of a specific type.
    """
    args_copy = args.copy()
    task_type = args_copy.pop('type')
    try:
        creation_func = task_creation_funcs[task_type]
        return creation_func(**args)
    except KeyError:
        raise ValueError(f'Unknown task type: {task_type}') from None
