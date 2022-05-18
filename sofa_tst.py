import Sofa.Core
import Sofa.constants.Key as Key
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile


class FingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self,args,kwargs)
        self.cable = args[0]
        self.name = "FingerController"

    def onKeypressedEvent(self, e):
        displacement = self.cable.CableConstraint.value[0]
        if e["key"] == Key.plus:
            displacement += 1.

        elif e["key"] == Key.minus:
            displacement -= 1.
            if displacement < 0:
                displacement = 0
        self.cable.CableConstraint.value = [displacement]


def Finger(parentNode=None, name="Finger",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
           fixingBox=[-5.0,0.0,0.0,10.0,15.0,20.0], pullPointLocation=[0.0,0.0,0.0]):

    finger = parentNode.addChild(name)
    eobject = ElasticMaterialObject(finger,
                                       volumeMeshFileName="data/finger.vtk",
                                       poissonRatio=0.3,
                                       youngModulus=18000,
                                       totalMass=0.5,
                                       surfaceColor=[0.0, 0.8, 0.7, 1.0],
                                       surfaceMeshFileName="data/finger.stl",
                                       rotation=rotation,
                                       translation=translation)
    finger.addChild(eobject)

    FixedBox(eobject, atPositions=fixingBox, doVisualization=True)

    cable=PullingCable(eobject,
                         "PullingCable",
                         pullPointLocation=pullPointLocation,
                         rotation=rotation,
                         translation=translation,
                         cableGeometry=loadPointListFromFile("data/cable.json"));

    eobject.addObject(FingerController(cable))

    CollisionMesh(eobject, name="CollisionMesh",
                  surfaceMeshFileName="data/finger.stl",
                  rotation=rotation, translation=translation,
                  collisionGroup=[1, 2])

    CollisionMesh(eobject, name="CollisionMeshAuto1",
                  surfaceMeshFileName="data/fingerCollision_part1.stl",
                  rotation=rotation, translation=translation,
                  collisionGroup=[1])

    CollisionMesh(eobject, name="CollisionMeshAuto2",
                  surfaceMeshFileName="data/fingerCollision_part2.stl",
                  rotation=rotation, translation=translation,
                  collisionGroup=[2])

    return finger


def createScene(rootNode):
    from stlib3.scene import MainHeader, ContactHeader

    MainHeader(rootNode, gravity=[0.0, -981.0, 0.0], plugins=["SoftRobots"])
    ContactHeader(rootNode, alarmDistance=4, contactDistance=3, frictionCoef=0.08)
    rootNode.VisualStyle.displayFlags = "showBehavior showCollisionModels"

    Finger(rootNode, translation=[1.0,0.0,0.0])
    return rootNode


def main():
    import Sofa.Gui

    SofaRuntime.importPlugin("SofaBaseMechanics")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    # Generate the root node
    root = Sofa.Core.Node("root")

    # Call the above function to create the scene graph
    createScene(root)

    # Once defined, initialization of the scene graph
    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    # Initialization of the scene will be done here
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()

if __name__ == "__main__":
    main()