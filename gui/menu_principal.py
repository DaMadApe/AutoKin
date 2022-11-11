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
        frame_botones.grid(column=1, row=1, rowspan=3,
                           sticky='nsew', padx=10, pady=(50,10))

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

        # Panel de información de selección actual
        frame_selec = ttk.Frame(self)
        frame_selec.grid(column=2, row=1, sticky='ns', pady=(10,30))

        # Datos de robot (y modelo) seleccionado
        titulo_robot = ttk.Label(frame_selec, text="Robot",
                                 font=(13))
        titulo_robot.grid(column=0, row=0, columnspan=2)

        self.label_robot = ttk.Label(frame_selec)
        self.label_robot.grid(column=0, row=1, sticky='w')

        self.boton_config = ttk.Button(frame_selec, text="Config...",
                                       command=self.configurar_robot,
                                       width=12)
        self.boton_config.grid(column=1, row=1, sticky='e')

        self.label_modelo = ttk.Label(frame_selec)
        self.label_modelo.grid(column=0, row=2, sticky='w')

        self.boton_modelos = ttk.Button(frame_selec, text="Ver modelos",
                                       command=self.parent.ver_modelos,
                                       width=12)
        self.boton_modelos.grid(column=1, row=2, sticky='e')

        for child in frame_selec.winfo_children():
            child.grid_configure(padx=10, pady=10)

        # Estado de componentes
        titulo_estado = ttk.Label(self, text="Estado",
                                  font=(13))
        titulo_estado.grid(column=2, row=2)#, columnspan=2)

        self.frame_status = ttk.Frame(self)
        self.frame_status.grid(column=2, row=3, sticky='ns')#, pady=(10,30))

        # Botón para actualizar pantalla (estado)
        boton_actualizar = ttk.Button(self,
                                      text="Actualizar",
                                      width=12,
                                      command=self.actualizar)
        boton_actualizar.grid(column=2, row=4, sticky='n',
                              pady=10)

        self.actualizar()

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(3, weight=2)
        frame_botones.columnconfigure(0, weight=1)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(4, weight=2)

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
        robot_cls = self.controlador.robot_reg_s.cls_id
        popups[robot_cls](self,
                          callback=self.controlador.config_robot,
                          robot=self.controlador.robot_s)

    def actualizar(self, *args):
        super().actualizar()
        robot_reg_s = self.controlador.robot_reg_s
        modelo_reg_s = self.controlador.modelo_reg_s

        # Actualizar vista de selección de robot y modelo
        if modelo_reg_s is not None:
            model_nom = modelo_reg_s.nombre
            model_cls = modelo_reg_s.cls_id
            self.label_modelo.config(text=f"{model_nom}  ({model_cls})")
            self.boton_entrenar['state'] = 'normal'
            self.boton_controlar['state'] = 'normal'
        else:
            self.label_modelo.config(text="Modelo: Sin seleccionar")
            self.boton_entrenar['state'] = 'disabled'
            self.boton_controlar['state'] = 'disabled'

        if robot_reg_s is not None:
            robot_nom = robot_reg_s.nombre
            robot_cls = robot_reg_s.cls_id
            self.label_robot.config(text=f"{robot_nom}  ({robot_cls})")
            self.boton_config['state'] = 'normal'
            self.boton_modelos['state'] = 'normal'
        else:
            self.label_robot.config(text="Sin seleccionar")
            self.label_modelo.config(text=" ")
            self.boton_config['state'] = 'disabled'
            self.boton_modelos['state'] = 'disabled'

        # Actualizar estado de conecciones de robot externo
        for widget in self.frame_status.winfo_children():
            widget.destroy()

        ext_status = self.controlador.get_ext_status()
        if ext_status is not None:
            for i, (sis, status) in enumerate(ext_status.items()):
                sys_label = ttk.Label(self.frame_status, text=sis)
                sys_label.grid(column=0, row=i, padx=10, pady=10, sticky='w')

                status_label = ttk.Label(self.frame_status)
                status_label.grid(column=1, row=i, padx=10, pady=10, sticky='w')
                if status:
                    status_label.config(text=" activo ")
                    status_label['style'] = 'Green.TLabel'
                else:
                    status_label.config(text=" inactivo ")
                    status_label['style'] = 'Red.TLabel'
        else:
            label = ttk.Label(self.frame_status, text='Sin selección de robot externo')
            label.grid(column=0, row=0, columnspan=2, padx=10, pady=10)


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaMenuPrincipal(root)
    root.mainloop()