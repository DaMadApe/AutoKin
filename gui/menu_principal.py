import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Pantalla
from gui.popups_config import Popup_config_ext, Popup_config_rtb, Popup_config_sofa


class PantallaMenuPrincipal(Pantalla):

    def __init__(self, parent):
        super().__init__(parent, titulo="AutoKin")

    def definir_elementos(self):

        # Botones izquierdos
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=1, row=1, sticky='nsew', padx=10, pady=(50,10))

        boton_seleccionar = ttk.Button(frame_botones, text="Seleccionar",
                                       width=20,
                                       command=self.seleccionar_robot)
        boton_seleccionar.grid(column=0, row=0)

        self.boton_entrenar = ttk.Button(frame_botones, text="Entrenar",
                                          width=20,
                                          command=self.entrenar)
        self.boton_entrenar.grid(column=0, row=1)

        self.boton_controlar = ttk.Button(frame_botones, text="Controlar",
                                          width=20,
                                          command=self.controlar)
        self.boton_controlar.grid(column=0, row=2)

        for child in frame_botones.winfo_children():
            child.grid_configure(padx=10, pady=15)

        # Panel de información derecho
        frame_status = ttk.Frame(self)
        frame_status.grid(column=2, row=1, sticky='ns')

        # Datos de robot (y modelo) seleccionado
        titulo_robot = ttk.Label(frame_status, text="Robot",
                                 font=(13))
        titulo_robot.grid(column=0, row=0, columnspan=2)

        self.label_robot = ttk.Label(frame_status)
        self.label_robot.grid(column=0, row=1, sticky='w')

        self.boton_config = ttk.Button(frame_status, text="Config...",
                                       command=self.configurar_robot,
                                       width=12)
        self.boton_config.grid(column=1, row=1, sticky='e')

        self.label_modelo = ttk.Label(frame_status)
        self.label_modelo.grid(column=0, row=2, sticky='w')

        self.boton_modelos = ttk.Button(frame_status, text="Ver modelos",
                                       command=self.parent.ver_modelos,
                                       width=12)
        self.boton_modelos.grid(column=1, row=2, sticky='e')

        # Estado de componentes
        titulo_estado = ttk.Label(frame_status, text="Estado",
                                  font=(13))
        titulo_estado.grid(column=0, row=3, columnspan=2)

        self.status_labels = []

        for i, sis in enumerate(["Medición posición",
                                 "Controlador robot"]):
            sys_label = ttk.Label(frame_status, text=sis)
            sys_label.grid(column=0, row=4+i)

            status_label = ttk.Label(frame_status)
            status_label.grid(column=1, row=4+i)
            self.status_labels.append(status_label)

        self.actualizar()

        for child in frame_status.winfo_children():
            child.grid_configure(padx=10, pady=10)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(3, weight=2)
        frame_botones.columnconfigure(0, weight=1)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(2, weight=2)

    def seleccionar_robot(self):
        self.parent.seleccionar_robot()

    def entrenar(self):
        self.parent.entrenar_robot()

    def controlar(self):
        self.parent.controlar_robot()

    def configurar_robot(self):
        popups = {"Externo" : Popup_config_ext,
                  "Sim. RTB" : Popup_config_rtb,
                  "Sim. SOFA" : Popup_config_sofa}
        robot_cls = self.controlador.robot_selec.cls_id
        popups[robot_cls](self,
                          callback=self.controlador.config_robot,
                          robot=self.controlador.robot_s)

    def actualizar(self, *args):
        super().actualizar()
        robot_selec = self.controlador.robot_selec
        modelo_selec = self.controlador.modelo_selec

        if modelo_selec is not None:
            model_nom = modelo_selec.nombre
            model_cls = modelo_selec.cls_id
            self.label_modelo.config(text=f"{model_nom}  ({model_cls})")
            self.boton_entrenar['state'] = 'normal'
            self.boton_controlar['state'] = 'normal'
        else:
            self.label_modelo.config(text="Modelo: Sin seleccionar")
            self.boton_entrenar['state'] = 'disabled'
            self.boton_controlar['state'] = 'disabled'

        if robot_selec is not None:
            robot_nom = robot_selec.nombre
            robot_cls = robot_selec.cls_id
            self.label_robot.config(text=f"{robot_nom}  ({robot_cls})")
            self.boton_config['state'] = 'normal'
            self.boton_modelos['state'] = 'normal'
        else:
            self.label_robot.config(text="Sin seleccionar")
            self.label_modelo.config(text=" ")
            self.boton_config['state'] = 'disabled'
            self.boton_modelos['state'] = 'disabled'

        status = self.controlador.get_ext_status()
        for i, label in enumerate(self.status_labels):
            if status[i]:
                label.config(text="  conectado ")
                label['style'] = 'Green.TLabel'
            else:
                label.config(text="desconectado ")
                label['style'] = 'Red.TLabel'


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaMenuPrincipal(root)
    root.mainloop()