import logging
import serial
import time

import torch

from ext_robot.NatNetClient import NatNetClient
from autokin.utils import RobotExecError


PORT = 'COM4'


class ExtInstance:
    """
    Para enviar comandos de actuación a motores del robot
    y leer datos de posición enviados por las cámaras.
    """
    def __init__(self):

        logging.info("Iniciando instancia externa")

        self.p_stack = []

        self.body_recv_semaphore = False
        self.body_pos = None

        self.connect_mcu()
        self.connect_cam()

    def connect_mcu(self):
        try:
            self.serialESP = serial.Serial(PORT, 115200, timeout=15)
        except serial.serialutil.SerialException:
            logging.info("Falló conexión con uC")
            self.serialESP = None
        else:
            logging.info(f"uC conectado en {PORT}")

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
            logging.info(f"NatNet recibiendo datos: {rigidBodyCount} cuerpos rìgidos")
            self.cam_client_connected = True # Brincar if y correr sólo esta línea?

    def recv_body(self, id, position, rotation):
        # called once per rigid body per frame
        if self.body_recv_semaphore:
            # logging.debug(f"Id: {id}, pos: {position}, rot: {rotation}")
            if id == 1: # Base del robot
                self.body_pos = position
            if id == 2 and self.body_pos is not None:
                xr, yr, zr = position
                xb, yb, zb = self.body_pos
                self.p_stack.append([xr-xb, yr-yb, zr-zb])

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

        p_out = torch.zeros(q.shape[0], 3)
        
        for i, q_i in enumerate(q):
            self.body_recv_semaphore = True

            try:
                self.send_q_esp(q_i)
            except serial.serialutil.SerialException:
                raise RobotExecError
            
            while not self.p_stack:
                pass
            self.body_recv_semaphore = False
            logging.debug(f"p_{i} promediado de {len(self.p_stack)} medidas")
            p_out[i] = torch.mean(torch.tensor(self.p_stack), dim=0)
            self.p_stack = []

        # Revisar si ejecución fue correcta
        mcu_status, cam_status = self.status()
        if not (mcu_status and cam_status):
            logging.error(f"Cliente desconectado (MCU: {mcu_status}, Cam: {cam_status})")
            raise RobotExecError(f"Cliente desconectado (MCU: {mcu_status}, Cam: {cam_status})")

        return p_out

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