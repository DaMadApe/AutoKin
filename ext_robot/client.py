import logging
import serial
import time

import torch

from ext_robot.NatNetClient import NatNetClient
from autokin.utils import alinear_datos, RobotExecError


PORT = 'COM4'
FRAMES_PER_SAMPLE = 50


class ExtInstance:
    """
    Para enviar comandos de actuación a motores del robot
    y leer datos de posición enviados por las cámaras.
    """
    def __init__(self):

        logging.info("Iniciando instancia externa")

        self.in_exec = False
        self.p_stack = []

        self.body_recv_semaphore = False

        self.connect_mcu()
        self.connect_cam()

    def connect_mcu(self):
        try:
            self.serialESP = serial.Serial(PORT, 115200, timeout=5)
        except serial.serialutil.SerialException:
            logging.info("Falló conexión con uC")
            self.serialESP = None
        else:
            logging.info("uC conectado en ", PORT)

    def connect_cam(self):
        self.cam_client_connected = False
        self.cam_client = NatNetClient()
        self.cam_client.newFrameListener = self.recv_frame
        self.cam_client.rigidBodyListener = self.recv_body
        try:
            self.cam_client.run()
        except OSError:
            logging.info("Falló conexión de NatNet")
        else:
            logging.info("Sockets de NatNet conectados")


    def test_mcu_connection(self):
        return self.serialESP is not None and self.serialESP.is_open

    def test_cam_connection(self):
        self.cam_client_connected = False
        time.sleep(0.1) # Esperar un frame para registrar estado de conexión
        return self.cam_client_connected

    def recv_frame(self, frameNumber, 
                   markerSetCount,unlabeledMarkersCount,
                   rigidBodyCount,skeletonCount,labeledMarkerCount,
                   timecode,timecodeSub,timestamp,
                   isRecording,trackedModelsChanged):
        if not self.cam_client_connected:
            logging.info("NatNet recibiendo datos")
            self.cam_client_connected = True # Brincar if y correr sólo esta línea?
        if not frameNumber % FRAMES_PER_SAMPLE:
            self.body_recv_semaphore = True
            # logging.debug( "Frame", frameNumber )
        # self.current_frame=frameNumber # para tener esa info en recv_body?

    def recv_body(self, id, position, rotation):
        # called once per rigid body per frame
        if self.body_recv_semaphore:
            # logging.debug(f"Id: {id}, pos: {position}, rot: {rotation}")
            if self.in_exec: 
                self.p_stack.append(list(position))
            self.body_recv_semaphore = False

    # def send_cam_command(self, command_str):
    #     self.cam_client.sendCommand(command=self.cam_client.NAT_REQUEST,
    #                                 commandStr=command_str,
    #                                 socket=self.cam_client.commandSocket,
    #                                 address=(self.cam_client.serverIPAddress,
    #                                          self.cam_client.commandPort))

    def send_q_list_esp(self, q: torch.Tensor):
        for q_i in q:
            self.send_q_esp(q_i)

    def send_q_esp(self, q: torch.Tensor): # -> ack # Saber cuando termina ejecución
        msg_pasos = ','.join(str(int(val)) for val in q)
        logging.debug(f"q: {q}, Mensaje: {msg_pasos}")
        if self.serialESP is not None:
            self.serialESP.write(msg_pasos.encode('ascii'))

            # Bloquear ejecución hasta que ESP envíe señal de fin
            logging.debug("Esperando confirmación")
            ack = self.serialESP.read(1)
            if len(ack) == 0: # Si pasa timeout de serial.read
                logging.error("No se recibió confirmación de ejecución del uC")
                raise RobotExecError("No se recibió confirmación de ejecución del uC")
            logging.debug("Confirmado")
        else:
            raise RobotExecError("uC desconectado")

    def fkine(self, q: torch.Tensor):
        self.in_exec = True
        self.send_q_list_esp(q)
        self.in_exec = False

        p = torch.tensor(self.p_stack)
        self.p_stack = []

        # Revisar si ejecución fue correcta
        mcu_status, cam_status = self.status()
        if mcu_status and cam_status:
            logging.info(f"Fin de ejecución: q.shape = {q.shape}, p.shape = {p.shape}")
        else:
            logging.error(f"Cliente desconectado (MCU: {mcu_status}, Cam: {cam_status})")
            raise RobotExecError(f"Cliente desconectado (MCU: {mcu_status}, Cam: {cam_status})")

        q_out, p_out = alinear_datos(q, p)
        return q_out, p_out

    def status(self):
        mcu_status = self.test_mcu_connection()
        cam_status = self.test_cam_connection()

        # Intentar reconexión de sistemas si se desconectaron
        if not mcu_status:
            logging.info("uC desconectado, reintentando")
            self.connect_mcu()
            mcu_status = self.test_mcu_connection()
            logging.info(f"Conexión uC: {mcu_status}")

        if not cam_status:
            logging.info("NatNet desconectado, reintentando")
            self.connect_cam()
            cam_status = self.test_cam_connection()
            logging.info(f"Conexión Cam: {mcu_status}")

        return (mcu_status, cam_status)


if __name__ == "__main__":
    client = ExtInstance()
    input()