import casadi as cs
import numpy as np
from horizon import misc_function as misc
from horizon import variables as horizon_var
from collections import OrderedDict
import pickle
import time
from typing import Union, Iterable

# from horizon.type_doc import BoundsDict

default_thread_map = 10


class Function:
    """
        Function of Horizon: generic function of SX values from CASADI.

        Notes:
            This function is an abstract representation of its projection over the nodes of the optimization problem.
            An abstract function gets internally implemented at each node, using the variables at that node.
    """

    def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list, nodes_array: np.ndarray, is_receding=False,
                 thread_map_num=None):
        """
        Initialize the Horizon Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            nodes_array: binary array specifying the nodes the function is active on
        """
        self.is_receding = is_receding
        self._f = f
        self._name = name
        self._nodes_array = nodes_array
        # todo isn't there another way to get the variables from the function g?
        self.vars = used_vars
        self.pars = used_pars


        temp_var_nodes = np.array(range(self._nodes_array.size))

        # IF RECEDING, it is important to define two concepts:
        # - function EXISTS: the function exists only on the nodes where ALL the variables of the function are defined.
        # - function is ACTIVE: the function can be activated/disabled on the nodes where it exists
        for var in self.vars:
            # getNodes() in a OffsetVariable returns the nodes of the base variable.
            # here I want the nodes where the variable is actually defined, so I need to consider also the offset
            # when getting the nodes:

            # #todo very bad hack to remove SingleVariables and SingleParameters
            if -1 in var.getNodes():
                continue
            var_nodes = np.array(var.getNodes()) - var.getOffset()
            temp_var_nodes = np.intersect1d(temp_var_nodes, var_nodes)
        for par in self.pars:
            if -1 in par.getNodes():
                continue
            par_nodes = np.array(par.getNodes()) - par.getOffset()
            temp_var_nodes = np.intersect1d(temp_var_nodes, par_nodes)

        self._var_nodes = temp_var_nodes
        self._var_nodes_array = misc.getBinaryFromNodes(self._nodes_array.size, self._var_nodes)

        # if the function is active (self._nodes_array) in some nodes where the variables it involves are not defined (self._var_nodes) throw an error.
        # this is true for the offset variables also: an offset variable of a variable defined on [0, 1, 2] is only valid at [1, 2].
        check_feas_nodes = self._var_nodes_array - self._nodes_array
        if (check_feas_nodes < 0).any():
            raise ValueError(f'Function "{self.getName()}" cannot be active on nodes: {np.where(check_feas_nodes < 0)}')

        if thread_map_num is None:
            self.thread_map_num = default_thread_map
        else:
            self.thread_map_num = thread_map_num

        # create function of CASADI, dependent on (in order) [all_vars, all_pars]
        all_input = self.vars + self.pars
        all_names = [i.getName() for i in all_input]

        self._fun = cs.Function(name, self.vars + self.pars, [self._f], all_names, ['f'])
        self._fun_impl = None
        self._project()

    def getName(self) -> str:
        """
        Getter for the name of the function

        Returns:
            name of the function
        """
        return self._name

    def getDim(self) -> int:
        """
        Getter for the dimension of the function

        Returns:
            dimension of the function
        """
        return self._f.shape[0]

    def getFunction(self) -> cs.Function:
        """
        Getter for the CASADI function

        Returns:
            instance of the CASADI function
        """
        return self._fun

    def getImpl(self, nodes=None):
        """
        Getter for the CASADI function implemented at the desired node
        Args:
            node: the desired node of the implemented function to retrieve

        Returns:
            instance of the CASADI function at the desired node
        """
        if self.is_receding:
            # if receding is True, always return a vector with the implemented function on all the nodes
            return cs.vertcat(*[self._fun_impl])

        if nodes is None:
            nodes = misc.getNodesFromBinary(self._nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._nodes_array)

        # otherwise I have to convert the input nodes to the corresponding column position:
        #     function active on [5, 6, 7] means that the columns are 0, 1, 2 so i have to convert, for example, 6 --> 1
        pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        # todo add guards
        # nodes = misc.checkNodes(nodes, self._nodes)
        # getting the column corresponding to the nodes requested
        fun_impl = cs.vertcat(*[self._fun_impl[:, pos_nodes]])

        return fun_impl

    def _getUsedElemImpl(self, elem_container):
        # todo throw with a meaningful error when nodes inserted are wrong

        used_elem_impl = list()
        for elem in elem_container:
            if self.is_receding:
                # get only the nodes of the function where all the variables of the function are defined:
                impl_nodes = self._var_nodes
            else:
                impl_nodes = self.getNodes()

            elem_impl = elem.getImpl(impl_nodes)
            used_elem_impl.append(elem_impl)
        return used_elem_impl

    def _getUsedVarImpl(self):
        return self._getUsedElemImpl(self.vars)

    def _getUsedParImpl(self):
        return self._getUsedElemImpl(self.pars)

    def _project(self):
        """
        Implements the function at the desired node using the desired variables.

        Args:
            used_vars: the variable used at the desired node

        Returns:
            the implemented function
        """
        if self.is_receding:
            # num_nodes = self._nodes_array.size
            # get only the nodes of the function where all the variables of the function are defined:
            num_nodes = self._var_nodes.size
        else:
            num_nodes = int(np.sum(self._nodes_array))

        if num_nodes == 0:
            # if the function is not specified on any nodes, don't implement
            self._fun_impl = None
        else:
            # mapping the function to use more cpu threads
            self._fun_map = self._fun.map(num_nodes, 'thread', self.thread_map_num)
            used_var_impl = self._getUsedVarImpl()
            used_par_impl = self._getUsedParImpl()
            all_vars = used_var_impl + used_par_impl
            fun_eval = self._fun_map(*all_vars)
            self._fun_impl = fun_eval

    def getNodes(self) -> list:
        """
        Getter for the active nodes of the function.

        Returns:
            a list of the nodes where the function is active

        """
        return misc.getNodesFromBinary(self._nodes_array)

    def setNodes(self, nodes, erasing=False):
        """
        Setter for the active nodes of the function.

        Args:
            nodes: list of desired active nodes.
            erasing: choose if the inserted nodes overrides the previous active nodes of the function. 'False' if not specified.
        """

        # todo this method is very important. It projects the abstract functions on the nodes specified using the implemented variables
        if erasing:
            self._nodes_array[:] = 0

        # adding to function nodes
        if self.is_receding:
            pos_nodes = nodes
        else:
            pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        self._nodes_array[pos_nodes] = 1

        # todo this is redundant. If the implemented variables do not change, this is not required, right?
        #   How do I understand when the var impl changed?
        # usually the number of nodes stays the same, while the active nodes of a function may change.
        # If the number of nodes changes, also the variables change. That is when this reprojection is required.
        self._project()

    def getVariables(self, offset=True) -> list:
        """
        Getter for the variables used in the function.

        Returns:
            a dictionary of all the used variable. {name: cs.SX or cs.MX}
        """

        if offset:
            ret = self.vars
        else:
            ret = list()
            for var in self.vars:
                if var.getOffset() == 0:
                    ret.append(var)

        return ret

    def getParameters(self) -> list:
        """
        Getter for the parameters used in the function.

        Returns:
            a dictionary of all the used parameters. {name: cs.SX or cs.MX}
        """
        return self.pars

    def getType(self) -> str:
        """
        Getter for the type of the Variable.

        Notes:
            Is it useful?

        Returns:
            a string describing the type of the function
        """
        return 'generic'

    # def __reduce__(self):
    #     """
    #     Experimental function to serialize this element.
    #
    #     Returns:
    #         instance of this element serialized
    #     """
    #     return (self.__class__, (self._name, self._f, self.vars, self.pars, self._nodes, ))

    def serialize(self):
        """
        Serialize the Function. Used to save it.

        Returns:
            serialized instance of Function
        """

        self._f = self._f.serialize()

        for i in range(len(self.vars)):
            self.vars[i] = self.vars[i].serialize()

        for node, item in self._fun_impl.items():
            self._fun_impl[node] = item.serialize()

        # self._fun = self._fun.serialize()

        return self

    def deserialize(self):
        """
        Deserialize the Function. Used to load it.

        Returns:
            deserialized instance of Function
        """

        self._f = cs.SX.deserialize(self._f)

        for i in range(len(self.vars)):
            self.vars[i] = cs.SX.deserialize(self.vars[i])

        for node, item in self._fun_impl.items():
            self._fun_impl[node] = cs.SX.deserialize(item)

        # self._fun = cs.Function.deserialize(self._fun)

        return self

# class RecedingFunction(Function):
#
#     def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list, nodes_array: np.ndarray,
#                  thread_map_num=None):
#
#         super().__init__(name, f, used_vars, used_pars, nodes_array, thread_map_num)


class Constraint(Function):
    """
    Constraint Function of Horizon.
    """

    def __init__(self, name: str, f: Union[cs.SX, cs.MX], used_vars: list, used_pars: list, nodes_array: np.ndarray,
                 bounds=None, is_receding=False, thread_map_num=None):
        """
        Initialize the Constraint Function.

        Args:
            name: name of the constraint function
            f: constraint SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            nodes_array: nodes the function is active on
            bounds: bounds of the constraint. If not specified, the bounds are set to zero.
        """

        super().__init__(name, f, used_vars, used_pars, nodes_array, is_receding, thread_map_num)
        self.bounds = dict()

        self.receding = is_receding
        # todo the bounds vector should be dim x active_nodes if not receding
        if is_receding:

            num_nodes = int(np.sum(self._var_nodes_array))
            temp_lb = -np.inf * np.ones([f.shape[0], num_nodes])
            temp_ub = np.inf * np.ones([f.shape[0], num_nodes])

            # this is zero only on the nodes where the function is ACTIVE (which are generally different from the nodes where the function EXISTS)
            active_nodes = np.where(self._nodes_array == 1)[0]
            pos_nodes = misc.convertNodestoPos(active_nodes, self._var_nodes_array)

            temp_lb[:, pos_nodes] = 0.
            temp_ub[:, pos_nodes] = 0.

            self.bounds['lb'] = temp_lb
            self.bounds['ub'] = temp_ub

        else:
        # constraints are initialize to 0.: 0. <= x <= 0.
            num_nodes = int(np.sum(nodes_array))
            self.bounds['lb'] = np.full((f.shape[0], num_nodes), 0.)
            self.bounds['ub'] = np.full((f.shape[0], num_nodes), 0.)



        # manage bounds
        if bounds is not None:
            if 'nodes' not in bounds:
                bounds['nodes'] = None

            if 'lb' in bounds:
                if 'ub' not in bounds:
                    bounds['ub'] = np.full(f.shape[0], np.inf)

            if 'ub' in bounds:
                if 'lb' not in bounds:
                    bounds['lb'] = np.full(f.shape[0], -np.inf)

            self.setBounds(lb=bounds['lb'], ub=bounds['ub'], nodes=bounds['nodes'])

    # todo transform string in typeFun "ConstraintType"
    def getType(self) -> str:
        """
        Getter for the type of the Constraint.

        Returns:
            a string describing the type of the function
        """
        return 'constraint'

    def _setVals(self, val_type, val, nodes=None):
        """
        Generic setter.

        Args:
            val_type: type of value
            val: desired values to set
            nodes: which nodes the values are applied on
        """
        if nodes is None:
            nodes = misc.getNodesFromBinary(self._nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._nodes_array)

        val_checked = misc.checkValueEntry(val)
        if val_checked.shape[0] != self.getDim():
            raise Exception('Wrong dimension of upper bounds inserted.')

        if self.receding:
            pos_nodes = misc.convertNodestoPos(nodes, self._var_nodes_array)
        else:
            pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        # for node in nodes:
        #     if node in self._nodes:
        # todo guards (here it is assumed that bounds is a row)
        val_type[:, pos_nodes] = val_checked

    def setLowerBounds(self, bounds, nodes=None):
        """
        Setter for the lower bounds of the function.

        Args:
            bounds: desired bounds of the function
            nodes: nodes of the function the bounds are applied on. If not specified, the function is bounded along ALL the nodes.
        """
        self._setVals(self.bounds['lb'], bounds, nodes)

    def setUpperBounds(self, bounds, nodes=None):
        """
        Setter for the upper bounds of the function.

        Args:
            bounds: desired bounds of the function
            nodes: nodes of the function the bounds are applied on. If not specified, the function is bounded along ALL the nodes.
        """
        self._setVals(self.bounds['ub'], bounds, nodes)

    def setBounds(self, lb, ub, nodes=None):
        """
        Setter for the bounds of the function.

        Args:
            lb: desired lower bounds of the function
            ub: desired upper bounds of the function
            nodes: nodes of the function the bounds are applied on. If not specified, the function is bounded along ALL the nodes.
        """
        self.setLowerBounds(lb, nodes)
        self.setUpperBounds(ub, nodes)

    def _getVals(self, val_type, nodes=None):
        """
        wrapper function to get the desired argument from the constraint.

        Args:
            val_type: type of the argument to retrieve
            node: desired node at which the argument is retrieved. If not specified, this returns the desired argument at all nodes.

        Returns:
            value/s of the desired argument
        """

        if self.receding:
            return val_type

        if nodes is None:
            nodes = misc.getNodesFromBinary(self._nodes_array)
        else:
            nodes = misc.checkNodes(nodes, self._nodes_array)

        pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        # todo what is this???
        if len(nodes) == 0:
            return np.zeros((self.getDim(), 0))

        vals = val_type[:, pos_nodes]

        return vals

    def getLowerBounds(self, node: int = None):
        """
        Getter for the lower bounds of the function.

        Args:
            node: desired node at which the lower bounds are retrieved. If not specified, this returns the lower bounds at all nodes.

        Returns:
            value/s of the lower bounds

        """
        lb = self._getVals(self.bounds['lb'], node)
        return lb

    def getUpperBounds(self, node: int = None):
        """
        Getter for the upper bounds of the function.

        Args:
            node: desired node at which the upper bounds are retrieved. If not specified, this returns the upper bounds at all nodes.

        Returns:
            value/s of the upper bounds

        """
        ub = self._getVals(self.bounds['ub'], node)
        return ub

    def getBounds(self, nodes=None):
        """
        Getter for the bounds of the function.

        Args:
            node: desired node at which the bounds are retrieved. If not specified, this returns the bounds at all nodes.

        Returns:
            value/s of the upper bounds
        """
        return self.getLowerBounds(nodes), self.getUpperBounds(nodes)

    def setNodes(self, nodes, erasing=False):
        """
        Setter for the active nodes of the constraint function.

        Args:
            nodes: list of desired active nodes.
            erasing: choose if the inserted nodes overrides the previous active nodes of the function. 'False' if not specified.
        """
        super().setNodes(nodes, erasing)

        if self.is_receding:
            pos_nodes = nodes
        else:
            pos_nodes = misc.convertNodestoPos(nodes, self._nodes_array)

        # for all the "new nodes" that weren't there, add default bounds
        self.bounds['lb'][:, pos_nodes] = np.zeros([self._f.shape[0], 1])
        self.bounds['ub'][:, pos_nodes] = np.zeros([self._f.shape[0], 1])

class CostFunction(Function):
    """
    Cost Function of Horizon.
    """

    def __init__(self, name, f, used_vars, used_pars, nodes_array, is_receding=False, thread_map_num=None):
        """
        Initialize the Cost Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            nodes_array: binary array specifying the nodes the function is active on
        """

        super().__init__(name, f, used_vars, used_pars, nodes_array, is_receding, thread_map_num)

        # if is_receding:
        #     self.weight_mask = None

    def setNodes(self, nodes, erasing=False):

        super().setNodes(nodes, erasing)

        if self.is_receding:
            # eliminate/enable cost functions by setting their weight
            nodes_mask = np.zeros([self.getDim(), np.zeros(int(np.sum(self._var_nodes_array)))])
            nodes_mask[:, nodes] = 1.
            self.weight_mask.assign(nodes_mask)

    def getType(self):
        """
        Getter for the type of the Cost Function.

        Returns:
            a string describing the type of the function
        """
        return 'costfunction'

    # def recede(self):


class ResidualFunction(Function):
    """
    Residual Function of Horizon.
    """

    def __init__(self, name, f, used_vars, used_pars, nodes_array, is_receding=False, thread_map_num=None):
        """
        Initialize the Residual Function.

        Args:
            name: name of the function
            f: SX function
            used_vars: variable used in the function
            used_pars: parameters used in the function
            nodes_array: binary array specifying the nodes the function is active on
        """
        super().__init__(name, f, used_vars, used_pars, nodes_array, is_receding, thread_map_num)

    def setNodes(self, nodes, erasing=False):

        super().setNodes(nodes, erasing)

        if self.is_receding:
            # eliminate/enable cost functions by setting their weight
            nodes_mask = np.zeros([self.getDim(), np.zeros(int(np.sum(self._var_nodes_array)))])
            nodes_mask[:, nodes] = 1.
            self.weight_mask.assign(nodes_mask)

    def getType(self):
        """
        Getter for the type of the Cost Function.

        Returns:
            a string describing the type of the function
        """
        return 'residualfunction'


class FunctionsContainer:
    """
    Container of all the function of Horizon.
    It is used internally by the Problem to get the abstract and implemented function.

    Methods:
        build: builds the container with the updated functions.
    """

    def __init__(self, receding, thread_map_num, logger=None):
        """
        Initialize the Function Container.

        Args:
            receding: activate receding horizon
            thread_map_num: number of cpu threads used when implementing casadi map
            logger: a logger reference to log data
        """
        self._logger = logger
        self.receding = receding
        self.thread_map_num = thread_map_num

        # containers for the constraint functions
        self._cnstr_container = OrderedDict()

        # containers for the cost functions
        self._cost_container = OrderedDict()

    def createConstraint(self, name, g, used_var, used_par, nodes_array, bounds):

        fun = Constraint(name, g, used_var, used_par, nodes_array, bounds,
                         is_receding=self.receding, thread_map_num=self.thread_map_num)
        self.addFunction(fun)

        return fun

    def createCost(self, name, j, used_var, used_par, nodes_array):

        fun = CostFunction(name, j, used_var, used_par, nodes_array,
                           is_receding=self.receding, thread_map_num=self.thread_map_num)
        self.addFunction(fun)

        return fun

    def createResidual(self, name, j, used_var, used_par, nodes_array):

        fun = ResidualFunction(name, j, used_var, used_par, nodes_array,
                               is_receding=self.receding, thread_map_num=self.thread_map_num)
        self.addFunction(fun)
        return fun

    def addFunction(self, fun: Function):
        """
        Add a function to the Function Container.

        Args:
            fun: a Function (can be Constraint or Cost Function) o add
        """
        # todo refactor this using types
        if fun.getType() == 'constraint':
            if fun.getName() not in self._cnstr_container:
                self._cnstr_container[fun.getName()] = fun
            else:
                raise Exception(f'Function name "{fun.getName()}" already inserted.')
        elif fun.getType() == 'costfunction':
            if fun.getName() not in self._cost_container:
                self._cost_container[fun.getName()] = fun
            else:
                raise Exception(f'Function name "{fun.getName()}" already inserted.')
        elif fun.getType() == 'residualfunction':
            if fun.getName() not in self._cost_container:
                self._cost_container[fun.getName()] = fun
            else:
                raise Exception(f'Function name "{fun.getName()}" already inserted.')
        elif fun.getType() == 'generic':
            print('functions.py: generic not implemented')
        else:
            raise Exception('Function type not implemented')

    def removeFunction(self, fun_name: str):
        """
        Remove a function from the Function Container.

        Args:
            fun_name: name of the function to be removed
        """
        if fun_name in self._cnstr_container:
            del self._cnstr_container[fun_name]
            return True
        elif fun_name in self._cost_container:
            del self._cost_container[fun_name]
            return True
        else:
            return False

    def getFunction(self, fun_name: str):
        """
        Getter for a Function inside the Function Container.
        Args:
            fun_name: name of the function to retrieve
        """
        if fun_name in self._cnstr_container:
            return self._cnstr_container[fun_name]
        elif fun_name in self._cost_container:
            return self._cost_container[fun_name]
        else:
            return None

    def getCnstr(self, name=None) -> OrderedDict:
        """
        Getter for the dictionary of all the abstract constraint functions.

        Args:
            name of constraint. If not specified, returns all of them
        Returns:
            ordered dict of the functions {name: fun}
        """
        if name is None:
            cnsrt_dict = self._cnstr_container
        else:
            if name in self._cnstr_container:
                cnsrt_dict = self._cnstr_container[name]
            else:
                cnsrt_dict = None
        return cnsrt_dict

    def getCost(self, name=None) -> OrderedDict:
        """
        Getter for the dictionary of all the abstract cost functions.

        Args:
            name of constraint. If not specified, returns all of them

        Returns:
            ordered dict of the functions {name: fun}
        """
        if name is None:
            cost_dict = self._cost_container
        else:
            cost_dict = self._cost_container[name]
        return cost_dict

    def getCnstrDim(self) -> int:
        """
        Getter for the dimension of all the constraints
        Returns:
            the total dimension of the constraint
        """
        total_dim = 0
        for cnstr in self._cnstr_container.values():
            total_dim += cnstr.getDim() * len(cnstr.getNodes())

        return total_dim

    def setNNodes(self, n_nodes):
        """
        set a desired number of nodes to Function Container.
        Args:
            n_nodes: the desired number of nodes to be set
        """

        # todo should find a way to understand which is the transcription function and project it over all the nodes
        # this is required to update the function_container EACH time a new number of node is set
        for cnstr in self._cnstr_container.values():
            # this is required to update the nodes consistently.
            # For instance, the horizon problem is specified on [0, 1, 2, 3, 4].
            # Consider a function containing an input variable. it is active on [0, 1, 2, 3].
            # I change the nodes to [0, 1, 2, 3]. The function must be updated accordingly: [0, 1, 2]
            # I change the nodes to [0, 1, 2, 3, 4, 5]. The function must be updated accordingly: [0, 1, 2, 4]
            available_nodes = set(range(n_nodes))
            # get only the variable it depends (not all the offsetted variables)
            for var in cnstr.getVariables(offset=False):
                if not var.getNodes() == [
                    -1]:  # todo very bad hack to check if the variable is a SingleVariable (i know it returns [-1]
                    available_nodes.intersection_update(var.getNodes())

            cnstr.setNodes([i for i in cnstr.getNodes() if i in available_nodes], erasing=True)

        for cost in self._cost_container.values():
            available_nodes = set(range(n_nodes))
            for var in cost.getVariables(offset=False):
                if not var.getNodes() == [-1]:
                    available_nodes.intersection_update(var.getNodes())
            cost.setNodes([i for i in cost.getNodes() if i in range(n_nodes)], erasing=True)

    def serialize(self):
        """
        Serialize the Function Container. Used to save it.

        Returns:
            instance of serialized Function Container

        """
        raise Exception('serialize yet to implement')
        for name, item in self._cnstr_container.items():
            self._cnstr_container[name] = item.serialize()

        for name, item in self._cost_container.items():
            self._cost_container[name] = item.serialize()

    def deserialize(self):
        """
        Deerialize the Function Container. Used to load it.

        Returns:
            instance of deserialized Function Container

        """
        raise Exception('serialize yet to implement')
        for name, item in self._cnstr_container.items():
            item.deserialize()
            new_vars = item.getVariables()
            for var in new_vars:
                print(var.getName(), var.getOffset())
                print(f'{item._f} depends on {var}?', cs.depends_on(item._f, var))

        exit()

        for name, item in self._cnstr_container.items():
            self._cnstr_container[name] = item.deserialize()

        # these are CASADI functions
        for name, item in self._cost_container.items():
            self._cost_container[name] = item.deserialize()


if __name__ == '__main__':
    x = cs.SX.sym('x', 2)
    y = cs.SX.sym('y', 2)
    fun = x + y
    used_var = dict(x=x, y=y)
    funimpl = Function('dan', fun, used_var, 1)

    funimpl = funimpl.serialize()
    print('===PICKLING===')
    funimpl_serialized = pickle.dumps(funimpl)
    print(funimpl_serialized)
    print('===DEPICKLING===')
    funimpl_new = pickle.loads(funimpl_serialized)
