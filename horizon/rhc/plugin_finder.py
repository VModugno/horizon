import importlib
import pkgutil
import json
import horizon.rhc.plugins
from horizon.rhc.tasks.task import Task
from horizon.rhc.tasks.cartesianTask import CartesianTask
from horizon.rhc.tasks.interactionTask import InteractionTask
from horizon.rhc.tasks.limitsTask import JointLimitsTask, VelocityLimitsTask
from horizon.rhc.tasks.posturalTask import PosturalTask

PLUGIN_NAME = 'pluginTry'
def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(horizon.rhc.plugins)
}


for i, k in discovered_plugins.items():
    print(f'{i}: {k}')

myplug = discovered_plugins[f'horizon.rhc.plugins.{PLUGIN_NAME}'].Penis('dio')

myplug.diocanepari()


# yaml file parser
# def main() -> None:
#     with open('./level.json') as file:
#         data = json.load(file)
#
#         tasks: list[Task] = []
#         for item in data['tasks']:
#             item_copy = item.copy()
#             task_type = item_copy.pop("type")
#             if task_type == 'Cartesian':
#                 tasks.append(CartesianTask(**item_copy))
#             elif task_type == 'Postural':
#                 tasks.append(PosturalTask(**item_copy))
#             elif
#                 ...
