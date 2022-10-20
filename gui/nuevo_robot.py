import tkinter as tk
from tkinter import ttk

import roboticstoolbox as rtb

from gui.gui_utils import Popup, Label_Entry
from autokin.robot import ExternRobot, RTBrobot, SofaRobot


class Popup_agregar_robot(Popup):

    robot_inits = {"Externo" : ExternRobot,
                   "Sim. RTB" : RTBrobot.from_name,
                   "Sim. SOFA" : SofaRobot}

    def __init__(self, parent, callback):
        self.callback = callback
        self.arg_getters = None
        super().__init__(title="Nuevo robot", parent=parent)

    def definir_elementos(self):
        # Entrada para el nombre del robot
        self.nom_entry = Label_Entry(self, label="Nombre", 
                                var_type='str', width=20)
        self.nom_entry.grid(column=0, row=0)

        # Selección de tipo de robot
        robot_cls_label = ttk.Label(self, text="Tipo de robot")
        robot_cls_label.grid(column=0, row=1)
        self.robot_cls_combo = ttk.Combobox(self,state='readonly')
        self.robot_cls_combo.grid(column=1, row=1)
        self.robot_cls_combo['values'] = tuple(self.robot_inits.keys())
        self.robot_cls_combo.bind('<<ComboboxSelected>>', self.definir_param_frame)

        # Frame para colocar configuraciones según tipo de robot
        self.param_frame = ttk.Frame(self)
        self.param_frame.grid(column=0, row=2, columnspan=2)

        # Mensajes de error
        self.label_error = ttk.Label(self.param_frame, text=' ')
        self.label_error['style'] = 'Red.TLabel'
        self.label_error.grid(column=0, row=3, columnspan=2)

        # Botones del fondo
        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.destroy)
        boton_cancelar.grid(column=0, row=4)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.agregar_robot)
        boton_aceptar.grid(column=1, row=4, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=3)

        self.bind('<Return>', self.agregar_robot)

    def definir_param_frame(self, event):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        self.arg_getters = {}
        tipo_robot = self.robot_cls_combo.get()

        if tipo_robot == "Externo":
            n_act_entry = Label_Entry(self.param_frame, label="# de actuadores",
                                  var_type='int', default_val=2,
                                  restr_positiv=True, non_zero=True)
            n_act_entry.grid(column=0, row=0)

            self.arg_getters['n'] = n_act_entry.get
            
        elif tipo_robot == "Sim. RTB":

            rtb_ref_label = ttk.Label(self.param_frame, text="Clave de robot")
            rtb_ref_label.grid(column=0, row=0)
            rtb_ref_combo = ttk.Combobox(self.param_frame,state='readonly')
            rtb_ref_combo.grid(column=1, row=0)
            rtb_model_list = rtb.models.DH.__all__
            rtb_ref_combo['values'] = rtb_model_list
            rtb_ref_combo.set(rtb_model_list[0])

            self.arg_getters['name'] = rtb_ref_combo.get

        elif tipo_robot == "Sim. SOFA":

            cable_configs = ('LS', 'LL', 'LSL', 'SLS',
                             'LLL', 'LSLS', 'LSSL', 'LLLL')

            cable_config_label = ttk.Label(self.param_frame, text="Config. cables")
            cable_config_label.grid(column=0, row=0)
            cable_config_combo = ttk.Combobox(self.param_frame,state='readonly')
            cable_config_combo.grid(column=1, row=0)
            cable_config_combo['values'] = cable_configs
            cable_config_combo.set(cable_configs[0])

            self.arg_getters['config'] = cable_config_combo.get

    def get_robot_kwargs(self) -> dict:
        robot_kwargs = {}
        for arg_name, get_fn in self.arg_getters.items():
            robot_kwargs[arg_name] = get_fn()
        return robot_kwargs

    def agregar_robot(self, *args):
        nombre = self.nom_entry.get()
        if self.arg_getters is not None and nombre != '':
            robot_cls = self.robot_cls_combo.get()
            robot_kwargs = self.get_robot_kwargs()
            if not (None in robot_kwargs.values()):

                robot_kwargs.update(cls_id=robot_cls)
                agregado = self.callback(nombre, robot_kwargs)

                if agregado:
                    self.destroy()
                else:
                    pass
                    #self.label_error.config(text='Nombre en uso')


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

    Popup_agregar_robot(root)
    root.mainloop()