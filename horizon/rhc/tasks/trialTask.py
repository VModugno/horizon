from horizon.rhc.tasks.task import Task
from horizon.variables import Variable
import casadi as cs
from horizon.problem import Problem
import numpy as np

class RegularizationTask(Task):

    @classmethod
    def from_description(cls, task_description):
        opt_variable = [] if 'variable' not in task_description else task_description['variable']


        if 'weight' in task_description and isinstance(task_description['weight'], dict):
            for var in task_description['weight'].keys():
                opt_variable.append(var)

        task = cls(opt_variable, **task_description)
        return task

    def __init__(self, opt_variable, task_node, context):
        super().__init__(task_node, context)

        self.opt_variable = opt_variable
        check_var = self.prb.getVariables(opt_variable.getName())

        if check_var is None:
            raise ValueError(f'variable {opt_variable} inserted is not in the problem.')

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createResidual

        self._initialize()

    def _initialize(self):
        self.reg_fun = self.instantiator(f'reg_{self.name}', self.weight * self.opt_variable, self.nodes)

    def setNodes(self, nodes):
        super().setNodes(nodes)
        self.reg_fun.setNodes(nodes, erasing=True)


# class RegularizationTaskInterface:
