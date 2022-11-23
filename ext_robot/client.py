import torch
import serial

from ext_robot.NatNetClient import NatNetClient
from autokin.utils import alinear_datos


class ExtInstance:
    """
    Para enviar comandos de actuación a motores del robot
    y leer datos de posición enviados por las cámaras.
    """
    def __init__(self):
        try:
            self.serialESP = serial.Serial("COM5",115200)
        except serial.serialutil.SerialException:
            self.serialESP = None

        self.cam_client = NatNetClient()
        self.cam_client_connected = False
        self.cam_client.newFrameListener = self.recv_frame
        self.cam_client.rigidBodyListener = self.recv_body
        self.cam_client.run()

        self.q_prev = [0]*4

        self.in_exec = False
        self.p_stack = []

    def test_connection(self, frameNumber, *args):
        self.connected = True

    def recv_frame(self, frameNumber, 
                   markerSetCount,unlabeledMarkersCount,
                   rigidBodyCount,skeletonCount,labeledMarkerCount,
                   timecode,timecodeSub,timestamp,
                   isRecording,trackedModelsChanged):
        if not self.cam_client_connected:
            self.cam_client_connected = True
        print( "Frame", frameNumber )
        # self.current_frame=frameNumber para tener esa info en recv_body?

    def recv_body(self, id, position, rotation):
        # called once per rigid body per frame
        print(f"Id: {id}, pos: {position}, rot: {rotation}")
        if self.in_exec: 
            # if not self.current_frame % 100: # Limitar ritmo al que se guardan muestras
            self.p_stack.append(list(position))

    def send_cam_command(self, command_str):
        self.cam_client.sendCommand(command=self.cam_client.NAT_REQUEST,
                                    commandStr=command_str,
                                    socket=self.cam_client.commandSocket,
                                    address=(self.cam_client.serverIPAddress,
                                             self.cam_client.commandPort))

    def send_q_list_esp(self, q: torch.tensor):
        for q_i in q:
            self.send_q_esp(q_i)

    def send_q_esp(self, q: torch.tensor): # -> ack # Saber cuando termina ejecución
        pasos = ['0']*4
        for i, val in enumerate(q):
            pasos[i] = str(int(val - self.q_prev[i]))
            self.q_prev[i] = val
        msg_pasos = ','.join(pasos)
        # print(f'Estado de robot: {self.q_prev}, Mensaje: {msg_pasos}')
        if self.serialESP is not None:
            self.serialESP.write(msg_pasos.encode('ascii'))
            # Bloquear ejecución hasta que ESP envíe señal de fin
            self.serialESP.read(1)

    def fkine(self, q: torch.tensor):
        # self.send_cam_command('StartRecording')
        self.in_exec = True
        self.send_q_list_esp(q)
        self.in_exec = False
        # self.send_cam_command('StopRecording')

        p = torch.tensor(self.p_stack)
        self.p_stack = []

        if all(self.status()):
            q_out, p_out = alinear_datos(q, p)
            return q_out, p_out
        else:
            print('Cliente desconectado: (MCU, Cam) = ', self.status())
            # HACK: Para que procesos posteriores terminen rápido sin error
            return torch.zeros(10, q.shape[1]), torch.zeros(10, 3)

    def status(self):
        mcu_status = self.serialESP is not None
        cam_status = self.cam_client_connected
        return (mcu_status, cam_status)


if __name__ == "__main__":
    client = ExtInstance()