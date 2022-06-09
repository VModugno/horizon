import yaml
import rospy
from cartesian_interface.pyci_all import *
from xbot_interface import xbot_interface as xbot
from xbot_interface import config_options as co

class CartesianUtils:
    def __init__(self, urdf, srdf, contact_names, dt):
        # ModelInterface and RobotStatePublisher
        self.model = self.init_model(urdf, srdf)
        self.__rspub = pyci.RobotStatePublisher(self.model)

        self.contact_names = contact_names
        self.ctrl_task = list()
        self.ik_dt = dt

        self.ci = self.make_cartesian_interface()

    def init_model(self, urdf, srdf):
        opt = co.ConfigOptions()
        opt.set_urdf(urdf)
        opt.set_srdf(srdf)
        opt.generate_jidmap()
        opt.set_bool_parameter('is_model_floating_base', True)
        opt.set_string_parameter('model_type', 'RBDL')
        opt.set_string_parameter('framework', 'ROS')
        model = xbot.ModelInterface(opt)

        q = model.getRobotState('home')
        model.setJointPosition(q)
        model.update()

        return model

    def make_problem_description(self):
        ik_cfg = dict()

        ik_cfg['solver_options'] = {'regularization': 1e-3, 'back_end': 'qpoases'}

        ik_cfg['stack'] = [
            self.contact_names + ['com'] + ['base_orientation'],
            ['postural']
        ]

        for contact in self.contact_names:
            ik_cfg[contact] = {
                'type': 'Cartesian',
                'base_link': 'world',
                'distal_link': contact,
                'indices': [0, 1, 2]
            }

        ik_cfg['com'] = {
            'type': 'Com'
        }

        ik_cfg['base_orientation'] = {
            'type': 'Cartesian',
            'base_link': 'world',
            'distal_link': 'base_link',
            'indices': [3, 4, 5]
        }

        ik_cfg['postural'] = {
            'type': 'Postural',
            'lambda': 0.1
        }

        ik_str = yaml.dump(ik_cfg)
        return ik_str

    def make_cartesian_interface(self):
        ik_pb = self.make_problem_description()
        log_path = '/tmp'
        ci = pyci.CartesianInterface.MakeInstance('OpenSot',
                                                  ik_pb,
                                                  self.model,
                                                  self.ik_dt,
                                                  log_path=log_path)

        # store the task list with the n contacts and com and postural after (useful?)
        for contact in self.contact_names:
            self.ctrl_task.append(ci.getTask(contact))

        return ci

    def setPoseReferences(self, ref_dict):
        contacts = list(ref_dict.keys())
        refs = list(ref_dict.values())

        for ref in ref_dict:
            if ref == 'Postural':
                self.ci.getTask(ref).setReferencePosture(self.model.eigenToMap(ref_dict[ref]))
            elif ref == 'base_link':
                self.ci.getTask(ref).setPoseReference(Affine3([0, 0, 0], ref_dict[ref]))
            else:
                self.ci.getTask(ref).setPoseReference(Affine3(ref_dict[ref]))

    def solve(self, t):
        if not self.ci.update(t, self.ik_dt):
            return False

        q = self.model.getJointPosition()
        qdot = self.model.getJointVelocity()

        q += qdot * self.ik_dt

        self.model.setJointPosition(q)
        self.model.update()

        print(q)

        return q

    def publishTransforms(self, prefix = ""):
        self.__rspub.publishTransforms(prefix)