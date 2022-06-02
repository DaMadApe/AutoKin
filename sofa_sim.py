import logging
from math import cos
from math import sin

import numpy as np

import Sofa.Core
import Sofa.constants.Key as Key
from splib3.numerics import Vec3, Quat
from splib3.animation import animate, AnimationManager

import os
path = os.path.dirname(os.path.abspath(__file__))+'/sofa_obj/'
dirPath = os.path.dirname(os.path.abspath(__file__))+'/'

class TrunkController(Sofa.Core.Controller):
    def __init__(self, trunk, *args, **kwargs):
        super().__init__(self,args,kwargs)
        self.trunk = trunk
        self.cable_type = 'L'
        self.cable_n = 0
        self.name = "TrunkController"

        q = 25 * np.load('q_in.npy')
        self.q = q #np.concatenate([np.zeros((100, q.shape[-1])),q])
        self.q_diff = np.diff(self.q, axis=0)

        self.step = 0
        self.p = np.zeros((len(self.q), 3))
        self.forces = np.zeros((len(self.q), 709)) # len(self.trunk.node.dofs.force.value)))

        self.update_selected_cable()

    def get_pos(self):
        return self.trunk.node.effector.mo.position[0]

    def onKeypressedEvent(self, e):
        displacement = self.cable.value[0]
        if e["key"] == Key.plus:
            displacement += 3.
            self.cable.value = [displacement]
            print(f'cable{self.cable_type}{self.cable_n} val: {self.cable.value[0]}')

        elif e["key"] == Key.minus:
            displacement -= 3.
            if displacement < 0:
                displacement = 0
            self.cable.value = [displacement]
            print(f'cable{self.cable_type}{self.cable_n} val: {self.cable.value[0]}')

        elif e["key"] == '.':
            self.cable_n += 1
            self.cable_n %= 4
            self.update_selected_cable()
        
        elif e["key"] == ',':
            self.cable_type = 'S' if self.cable_type=='L' else 'L'
            self.update_selected_cable()

        print(f'finger: {self.get_pos()}')

    def update_selected_cable(self):
        cable_handle = f'cable{self.cable_type}{self.cable_n}'
        self.cable = getattr(self.trunk.node, cable_handle).cable
        print(f'Cambio a {cable_handle}')

    def update_q(self, diff):
        for i, dq in enumerate(diff):
            cable = getattr(self.trunk.node, f'cableL{i}').cable
            cable.value += dq
            # print(f'cable{i}.value: {cable.value[0]}, {dq}')

    def onAnimateBeginEvent(self, event): # called at each begin of animation step
        self.step +=1
        if self.step < len(self.q) - 1:
            self.update_q(self.q_diff[self.step])
        # print(self.step)

    def onAnimateEndEvent(self, event):
        if self.step < len(self.q):
            self.p[self.step] = self.get_pos()

            forces = self.trunk.node.dofs.force.value
            forces = np.linalg.norm(forces, axis=1)
            self.forces[self.step] = forces
            # print(f'Efector final: {pos}')
            # print(self.p)
        elif self.step == len(self.q):
            self.p *= 0.1
            np.save(os.path.join(dirPath, 'p_out.npy'), self.p[1:])
            np.save(os.path.join(dirPath, 'forces_out.npy'), self.forces)


class Trunk():
    ''' This prefab is implementing a soft robot inspired by the elephant's trunk.
        The robot is entirely soft and actuated with 8 cables.
        The prefab is composed of:
        - a visual model
        - a collision model
        - a mechanical model for the deformable structure
        The prefab has the following parameters:
        - youngModulus
        - poissonRatio
        - totalMass
        Example of use in a Sofa scene:
        def createScene(root):
            ...
            trunk = Trunk(root)
            ## Direct access to the components
            trunk.displacements = [0., 0., 0., 0., 5., 0., 0., 0.]
    '''

    def __init__(self, parentNode, cableL_angs=None, cableS_angs=None,
                 youngModulus=450, poissonRatio=0.45, totalMass=0.042):

        self.node = parentNode.addChild('Trunk')

        self.node.addObject('MeshVTKLoader', name='loader', filename=path+'trunk.vtk')
        self.node.addObject('MeshTopology', src='@loader', name='container')

        self.node.addObject('MechanicalObject', name='dofs', template='Vec3')
        self.node.addObject('UniformMass', totalMass=totalMass)
        self.node.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=poissonRatio,  youngModulus=youngModulus)

        self.addCollisionModel()
        self.__addCables(cableL_angs, cableS_angs)
        self.addEffector()

    def __addCables(self, cableL_angs=None, cableS_angs=None):
        length1 = 10.
        length2 = 2.
        lengthTrunk = 195.

        if cableL_angs is None: 
            cableL_angs = [np.pi*i/2 for i in range(4)]

        if cableS_angs is None: 
            cableS_angs = [np.pi*i/2 for i in range(4)]

        pullPointsL = [[np.sin(t), np.cos(t), 0] for t in cableL_angs]
        pullPointsS = [[np.sin(t), np.cos(t), 0] for t in cableS_angs]

        direction = Vec3(0., length2-length1, lengthTrunk)
        direction.normalize()

        # Adición de cables largos
        for i in range(4):
            theta = cableL_angs[i]
            q = Quat(0., 0., sin(theta/2.), cos(theta/2.))

            position = [[0., 0., 0.]]*20
            for k in range(0, 20, 2):
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27)
                position[k+1] = v.rotateFromQuat(q)

            cableL = self.node.addChild(f'cableL{i}')
            cableL.addObject('MechanicalObject', name='dofs',
                                position=pullPointsL[i]+[pos.toList() for pos in position])
            cableL.addObject('CableConstraint', template='Vec3', name='cable',
                                hasPullPoint='0',
                                indices=list(range(0, 21)),
                                maxPositiveDisp='70',
                                maxDispVariation='1',
                                minForce=0)
            cableL.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

        # Adición de cables cortos
        for i in range(0, 4):
            theta = cableS_angs[i]
            q = Quat(0., 0., sin(theta/2.), cos(theta/2.))

            position = [[0., 0., 0.]]*10
            for k in range(0, 9, 2):
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27)
                position[k+1] = v.rotateFromQuat(q)

            cableS = self.node.addChild(f'cableS{i}')
            cableS.addObject('MechanicalObject', name='dofs',
                                position=pullPointsS[i]+[pos.toList() for pos in position])
            cableS.addObject('CableConstraint', template='Vec3', name='cable',
                                hasPullPoint='0',
                                indices=list(range(0, 10)),
                                maxPositiveDisp='40',
                                maxDispVariation='1',
                                minForce=0)
            cableS.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    def addVisualModel(self, color=[1., 1., 1., 1.]):
        trunkVisu = self.node.addChild('VisualModel')
        trunkVisu.addObject('MeshSTLLoader', filename=path+'trunk.stl')
        trunkVisu.addObject('OglModel', color=color)
        trunkVisu.addObject('BarycentricMapping')

    def addCollisionModel(self, selfCollision=False):
        trunkColli = self.node.addChild('CollisionModel')
        for i in range(2):
            part = trunkColli.addChild(f'Part{i+1}')
            part.addObject('MeshSTLLoader', name='loader', filename=path+'trunk_colli'+str(i+1)+'.stl')
            part.addObject('MeshTopology', src='@loader')
            part.addObject('MechanicalObject')
            part.addObject('TriangleCollisionModel', group=1 if not selfCollision else i)
            part.addObject('LineCollisionModel', group=1 if not selfCollision else i)
            part.addObject('PointCollisionModel', group=1 if not selfCollision else i)
            part.addObject('BarycentricMapping')

    def fixExtremity(self):
        self.node.addObject('BoxROI', name='boxROI', box=[[-20, -20, 0], [20, 20, 20]], drawBoxes=False)
        self.node.addObject('PartialFixedConstraint', fixedDirections=[1, 1, 1], indices='@boxROI.indices')

    def addEffector(self, position=[0., 0., 195.]):
        effectors = self.node.addChild('effector')
        effectors.addObject('MechanicalObject', name='mo',
                            position=position)
        effectors.addObject('BarycentricMapping', mapForces=False, mapMasses=False)


def createScene(rootNode):

    # logging.getLogger().setLevel(logging.ERROR)

    rootNode.addObject('RequiredPlugin', pluginName=['SoftRobots',
                                                     'SofaSparseSolver',
                                                     'SofaPreconditioner',
                                                     'SofaPython3',
                                                     'SofaConstraint',
                                                     'SofaImplicitOdeSolver',
                                                     'SofaLoader',
                                                     'SofaSimpleFem',
                                                     'SofaBoundaryCondition',
                                                     'SofaEngine',
                                                     'SofaOpenglVisual'])
    AnimationManager(rootNode)
    rootNode.addObject('VisualStyle', displayFlags='showBehavior')
    rootNode.gravity = [0., 0., 9810.] # [-9810., 0., 0.] # [0., -9810., 0.]

    rootNode.addObject('FreeMotionAnimationLoop')
    # For direct resolution, i.e direct control of the cable displacement
    rootNode.addObject('GenericConstraintSolver', maxIterations=100, tolerance=1e-5)

    simulation = rootNode.addChild('Simulation')

    simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
    simulation.addObject('ShewchukPCGLinearSolver', name='linearSolver', iterations=500, tolerance=1.0e-18, preconditioners='precond')
    simulation.addObject('SparseLDLSolver', name='precond')
    simulation.addObject('GenericConstraintCorrection', solverName='precond')

    trunk = Trunk(simulation, cableL_angs=[0, np.pi*2/3, 0, 0],
                              cableS_angs=[np.pi/3, 0, 0, 0])
    trunk.addVisualModel(color=[1., 1., 1., 0.8])
    trunk.fixExtremity()

    trunk.node.addObject(TrunkController(trunk)) # L=2, S=1

# export PYTHONPATH="/home/damadape/SOFA_robosoft/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH"
# export SOFA_ROOT="/home/damadape/SOFA_robosoft"
# ~/SOFA_robosoft/bin/runSofa '/home/damadape/Documents/Autokin/sofa_tst.py'