import phase_manager.pymanager as pymanager
import phase_manager.pyphase as pyphase

from horizon.rhc.taskInterface import TaskInterface
from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn

import casadi as cs

import numpy as np
import rospkg

import time

'''
1. phase manager should erase item's nodes (or at least handle them) after adding the item
    right now, if the constraint starts with all nodes ON, addingPhase() won't erase item in nodes were it is not active
'''
if __name__ == '__main__':

    cogimon_urdf_folder = rospkg.RosPack().get_path('cogimon_urdf')
    urdf = open(cogimon_urdf_folder + '/urdf/cogimon.urdf', 'r').read()

    ns = 10
    dt = 0.01
    prb = Problem(ns, receding=True, casadi_type=cs.SX)
    prb.setDt(dt)

    base_init = np.atleast_2d(np.array([0.03, 0., 0.962, 0., -0.029995, 0.0, 0.99955]))
    # base_init = np.array([0., 0., 0.96, 0., 0.0, 0.0, 1.])

    q_init = {"LHipLat": -0.0,
              "LHipSag": -0.363826,
              "LHipYaw": 0.0,
              "LKneePitch": 0.731245,
              "LAnklePitch": -0.307420,
              "LAnkleRoll": 0.0,
              "RHipLat": 0.0,
              "RHipSag": -0.363826,
              "RHipYaw": 0.0,
              "RKneePitch": 0.731245,
              "RAnklePitch": -0.307420,
              "RAnkleRoll": -0.0,
              "WaistLat": 0.0,
              "WaistYaw": 0.0,
              "LShSag": 1.1717860,
              "LShLat": -0.059091562,
              "LShYaw": -5.18150657e-02,
              "LElbj": -1.85118,
              "LForearmPlate": 0.0,
              "LWrj1": -0.523599,
              "LWrj2": -0.0,
              "RShSag": 1.17128697,
              "RShLat": 6.01664139e-02,
              "RShYaw": 0.052782481,
              "RElbj": -1.8513760,
              "RForearmPlate": 0.0,
              "RWrj1": -0.523599,
              "RWrj2": -0.0}

    # contact_dict = {
    #     'l_sole': {'type': 'surface'},
    #     'r_sole': {'type': 'surface'}
    # }

    contact_dict = {
        'l_sole': {
            'type': 'vertex',
            'vertex_frames': [
                'l_foot_lower_left_link',
                'l_foot_upper_left_link',
                'l_foot_lower_right_link',
                'l_foot_upper_right_link',
            ]
        },

        'r_sole': {
            'type': 'vertex',
            'vertex_frames': [
                'r_foot_lower_left_link',
                'r_foot_upper_left_link',
                'r_foot_lower_right_link',
                'r_foot_upper_right_link',
            ]
        }
    }

    kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
    model = FullModelInverseDynamics(problem=prb,
                                     kd=kin_dyn,
                                     q_init=q_init,
                                     base_init=base_init,
                                     contact_dict=contact_dict)


    ti = TaskInterface(prb, model=model)

    task_1_dict = {
                   'type': 'Cartesian',
                   'distal_link': 'base_link',
                   'name': 'final_base_x',
                   'indices': [0],
                   'nodes': 'all'
                   }

    task_1 = ti.setTaskFromDict(task_1_dict)

    opts = dict()
    opts['type'] = 'ipopt'

    ti.setSolverOptions(opts)
    ti.finalize()
    # =========================================================================
    # =========================================================================
    # =========================================================================

    pm = pymanager.PhaseManager(ns)
    # phase manager handling

    timeline_1 = pm.addTimeline(f'timeline_1')


    phase_1_duration = 5
    ref_trj = np.zeros(shape=[7, phase_1_duration])
    ref_trj[3, :] = [10., 10., 10., 10., 10.]
    phase_1 = pyphase.Phase(phase_1_duration, 'phase_1')
    phase_1.addItemReference(task_1, ref_trj)

    timeline_1.registerPhase(phase_1)


    # phase_2_duration = 3
    # phase_2 = pyphase.Phase(phase_2_duration, 'phase_2')
    # phase_2.addItem(cnsrt_1)

    # timeline_1.registerPhase(phase_2)
    print('=============== running ===============')

    timeline_1.addPhase(phase_1)
    # timeline_1.addPhase(phase_2)

    # for elem in timeline_1.getActivePhases():
    #     print(elem.getName())



    print('=============== results ===============')
    print(f"{task_1.getName()}'s nodes: {task_1.getNodes()}")
    for name, parameter in prb.getParameters().items():
        print(f"{name}: \n {parameter.getValues()}")

    print("=========> shifting once")
    tic = time.time()
    pm._shift_phases()
    # pm._shift_phases()
    # pm._shift_phases()
    # pm._shift_phases()

    print('=============== results ===============')
    print(f"{task_1.getName()}'s nodes: {task_1.getNodes()}")
    for name, parameter in prb.getParameters().items():
        print(f"{name}: \n {parameter.getValues()}")