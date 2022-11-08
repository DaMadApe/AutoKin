from ext_robot.control_motor import MCUServer
from ext_robot.NatNetClient import NatNetClient


def suavizar(q):
    return q

def alinear_datos(q, p):
    return q, p


class ExtInstance:
    """
    Para enviar comandos de actuación a motores del robot
    y leer datos de posición enviados por las cámaras.
    """
    def __init__(self):
        self.mcu_server = MCUServer()
        self.mcu_server.connect()

        self.cam_client = NatNetClient()
        self.cam_client.newFrameListener = self.recv_frame
        self.cam_client.rigidBodyListener = self.recv_body
        self.cam_client.run() # En el mismo lugar que inicia SofaInstance

        self.units = self.send_command_str('UnitsToMillimeters')
        self.frame_rate = self.send_command_str('FrameRate')

        self.in_exec = False # Inecesario si se puede iniciar/detener stream
        self.p_stack = []

    def recv_frame(self, frameNumber, 
                   markerSetCount,unlabeledMarkersCount,
                   rigidBodyCount,skeletonCount,labeledMarkerCount,
                   timecode,timecodeSub,timestamp,
                   isRecording,trackedModelsChanged):
        print( "Frame", frameNumber )

    def recv_body(self, id, position, rotation):
        # called once per rigid body per frame
        print(f"Id: {id}, pos: {position}, rot: {rotation}")
        if self.in_exec:
            self.p_stack.append(position)

    def send_command_str(self, command_str):
        self.cam_client.sendCommand(command=self.cam_client.NAT_MESSAGESTRING,
                                    commandStr=command_str,
                                    # Modificar NatNetClient o crear fun
                                    # diferente para facilitar esta llamada
                                    socket=self.cam_client.comm_socket,
                                    address=(self.cam_client.serverIPAddress,
                                            self.cam_client.commandPort))

    def fkine(self, q):
        self.send_command_str('StartRecording')
        self.in_exec = True
        q_ext = suavizar(q)
        self.mcu_server.send(q)
        self.mcu_server.wait_exec() # Esperar mensaje de vuelta
        self.send_command_str('StopRecording')
        q_out, p_out = alinear_datos(q_ext, self.p_stack)
        self.in_exec = False
        self.p_stack = []

        return q_out, p_out