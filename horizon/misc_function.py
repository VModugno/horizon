import numpy as np

def unravelElements(elements):
    if isinstance(elements, int):
        unraveled_elem = [elements]
        pass
    elif any(isinstance(el, list) for el in elements):
        unraveled_elem = list()
        for el in elements:
            temp = list(range(el[0], el[1]+1)) # +1 # todo cannot add+1 here?
            for item in temp:
                unraveled_elem.append(item) if item not in unraveled_elem else unraveled_elem
    elif isinstance(elements, list):
        unraveled_elem = list()
        temp = list(range(elements[0], elements[1]+1)) # +1
        for item in temp:
            unraveled_elem.append(item)

    return unraveled_elem


def listOfListFLOATtoINT(listOfList):
    # transform every element to INT
    for i in range(len(listOfList)):
        if isinstance(listOfList[i], list):
            for j in range(len(listOfList[i])):
                listOfList[i][j] = int(listOfList[i][j])
        else:
            listOfList[i] = int(listOfList[i])

    return listOfList

def checkNodes(nodes, nodes_array=None):

    nodes_vec = np.array(nodes)
    # todo check for repeated nodes
    if nodes_array is None:
        checked_nodes = nodes_vec
    else:
        # get from nodes only the nodes active in nodes_array
        checked_nodes = np.ma.masked_array(nodes_vec, mask=np.logical_not(nodes_array[nodes_vec])).compressed()

    # if hasattr(nodes, "__iter__") and not isinstance(nodes, str):
    #     if nodes_array is None:
    #         checked_nodes = nodes
    #     else:
    #
    #         checked_nodes = nodes_array[nodes]
    #         np.array()
    #
    #         print(checked_nodes)
    #         exit()
    # elif isinstance(nodes, (int, np.integer)):
    #     if nodes_array is None:
    #         checked_nodes = [nodes]
    #     else:
    #         checked_nodes = [nodes] if nodes in nodes_array else []
    # else:
    #     raise Exception('Type {} is not supported to specify nodes.'.format(type(nodes)))
    #
    # if checked_nodes is None:
    #     raise Exception(f'Invalid nodes inserted: {nodes}')
    # # todo ADD WARNING if node (or nodes) are NOT present in own nodes.
    # #  (basically, when setting bounds for some node, it is left out because the var/fun does not exist in that node
    return checked_nodes

def checkValueEntry(val):
    if isinstance(val, (int, float)):
        val = np.array([val])
    else:
        val = np.array(val)

        # if single value, flatten
        # note: dont flatten matrix of values!
        multiple_vals = val.ndim == 2 and val.shape[1] != 1
        if not multiple_vals:
            val = val.flatten()

    return val

def getBinaryFromNodes(total_nodes: int, active_nodes: list):
    # add guards
    # if not isinstance(nodes, list):
    #     raise TypeError('input must be a list of nodes')

    nodes_array = np.zeros(total_nodes)
    nodes_array[active_nodes] = 1

    return nodes_array

def getNodesFromBinary(nodes_array):
    # add guards
    # if not isinstance(nodes_array, list):
    #     raise TypeError('input must be a list of binary nodes')
    nodes = np.where(nodes_array == 1)[0]

    return nodes

if __name__ == '__main__':

    print(checkNodes([3,3,4,5,6], np.array([0, 0, 1, 1, 1, 0, 0, 1, 0])))