import tkinter as tk
from tkinter import ttk

from gui.robot_database import UIController


class PantallaMenuPrincipal(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title("AutoKin")

        self.parent = parent
        self.controlador = UIController()

        self.definir_elementos()

    def definir_elementos(self):

        style= ttk.Style()
        style.configure('Green.TLabel', foreground='#3A2')
        style.configure('Red.TLabel', foreground='#A11')

        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=1, row=1, sticky='nsew', padx=10, pady=(50,10))

        # Botones izquierdos
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
                                       command=self.config_robot,
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

        self.label_sys1 = ttk.Label(frame_status, text="Medición posición")
        self.label_sys1.grid(column=0, row=4)

        self.label_status1 = ttk.Label(frame_status)
        self.label_status1.grid(column=1, row=4)

        self.label_sys2 = ttk.Label(frame_status, text="Controlador robot")
        self.label_sys2.grid(column=0, row=5)

        self.label_status2 = ttk.Label(frame_status)
        self.label_status2.grid(column=1, row=5)

        self.actualizar_status()
        # self.bind("<FocusIn>", self.actualizar_status)
        # self.bind("<FocusOut>", self.actualizar_status)

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

    def config_robot(self):
        pass

    def actualizar_status(self, *args):
        self.controlador.cargar()
        robot_selec = self.controlador.robot_selec()
        modelo_selec = self.controlador.modelo_selec()

        if modelo_selec is not None:
            model_nom = modelo_selec.nombre
            model_cls = modelo_selec.modelo.hparams['tipo']
            self.label_modelo.config(text=f"{model_nom}({model_cls})")
            self.boton_entrenar['state'] = 'normal'
            self.boton_controlar['state'] = 'normal'
        else:
            self.label_modelo.config(text="Modelo: Sin seleccionar")
            self.boton_entrenar['state'] = 'disabled'
            self.boton_controlar['state'] = 'disabled'

        if robot_selec is not None:
            robot_nom = robot_selec.nombre
            robot_cls = robot_selec.robot.__class__.__name__
            self.label_robot.config(text=f"{robot_nom}  ({robot_cls})")
            self.boton_config['state'] = 'normal'
            self.boton_modelos['state'] = 'normal'
        else:
            self.label_robot.config(text="Sin seleccionar")
            self.label_modelo.config(text=" ")
            self.boton_config['state'] = 'disabled'
            self.boton_modelos['state'] = 'disabled'

        # TODO: if self.controlador.get_status():
        self.label_status1.config(text="  conectado ")
        self.label_status1['style'] = 'Green.TLabel'

        self.label_status2.config(text="desconectado")
        self.label_status2['style'] = 'Red.TLabel'


if __name__ == '__main__':

    root = tk.Tk()
    root.minsize(550,330)
    root.maxsize(1200,800)

    win_width = 800
    win_height = 450
    x_pos = int(root.winfo_screenwidth()/2 - win_width/2)
    y_pos = int(root.winfo_screenheight()/2 - win_height/2)

    geom = f'{win_width}x{win_height}+{x_pos}+{y_pos}'
    root.geometry(geom)

    pant1 = PantallaMenuPrincipal(root)
    root.mainloop()