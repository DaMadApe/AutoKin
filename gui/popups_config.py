from functools import partial
import tkinter as tk
from tkinter import ttk

from autokin.robot import ExternRobot, RTBrobot, SofaRobot
from gui.gui_utils import Popup, Label_Entry


class Popup_config_ext(Popup):

    def __init__(self, parent, callback, robot: ExternRobot):
        self.callback = callback
        self.robot = robot
        super().__init__(title="Configurar robot: Externo", parent=parent)

    def definir_elementos(self):
        pass


class Popup_config_rtb(Popup):

    def __init__(self, parent, callback, robot: RTBrobot):
        self.callback = callback
        self.robot = robot
        super().__init__(title="Configurar robot: RTB", parent=parent)

    def definir_elementos(self):
        pass


class Popup_config_sofa(Popup):

    def __init__(self, parent, callback, robot: SofaRobot):
        self.callback = callback
        self.robot = robot
        self.min_vars = []
        self.max_vars = []

        self.old_config = {'q_min': robot.q_min.tolist(),
                           'q_max': robot.q_max.tolist(),
                           'headless': robot.headless}
        super().__init__(title="Configurar robot: Sofa", parent=parent)

        self.robot.headless = False
        self.robot.start_instance()

    def definir_elementos(self):
        # Configuración de mínimos/máximos de actuación
        frame_spins = ttk.Frame(self)
        frame_spins.grid(column=0, row=0)

        self.protocol("WM_DELETE_WINDOW", self.cancelar)

        for i in range(self.robot.n):
            q_label = ttk.Label(frame_spins, text=f"q{i+1}")
            q_label.grid(column=0, row=i)

            min_var = tk.DoubleVar(value=float(self.robot.q_min[i]))
            self.min_vars.append(min_var)
            max_var = tk.DoubleVar(value=float(self.robot.q_max[i]))
            self.max_vars.append(max_var)

            min_spin = ttk.Spinbox(frame_spins, width=5,
                                   from_=0.0, to=50.0, increment=0.5,
                                   textvariable=min_var, 
                                   command=partial(self.min_spin_command, i))
            min_spin.grid(column=1, row=i)

            max_spin = ttk.Spinbox(frame_spins, width=5,
                                   from_=0.0, to=50.0, increment=0.5,
                                   textvariable=max_var, 
                                   command=partial(self.max_spin_command, i))
            max_spin.grid(column=3, row=i)

            min_but = ttk.Button(frame_spins, text="Set min",
                                 width=7,
                                 command=partial(self.set_min, i))
            min_but.grid(column=2, row=i)

            max_but = ttk.Button(frame_spins, text="Set max",
                                 width=7,
                                 command=partial(self.set_max, i))
            max_but.grid(column=4, row=i)

        # Definir factor de escala para medidas de posición
        frame_entry = ttk.Frame(self)
        frame_entry.grid(column=0, row=1, sticky='w')

        self.p_scale_entry = Label_Entry(frame_entry, label="Escala de posiciones", 
                                         var_type='float', width=5)
        self.p_scale_entry.grid(column=0, row=0)
        self.p_scale_entry.set(self.robot.p_scale)
 
        # Botones del fondo
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=0, row=2)

        self.check_var = tk.IntVar(value=self.robot.headless)
        check_tens = ttk.Checkbutton(frame_botones,
                                     text='Usar interfaz gráfica',
                                     variable=self.check_var)
        check_tens.grid(column=0, row=1, columnspan=3, sticky='w')

        jog_but = ttk.Button(frame_botones, text="Jog",
                             width=12,
                             command=lambda: self.jog)
        jog_but.grid(column=0, row=2)

        aceptar_but = ttk.Button(frame_botones, text="Aceptar",
                                 width=12,
                                 command=self.aceptar)
        aceptar_but.grid(column=1, row=2)

        cancelar_but = ttk.Button(frame_botones, text="Cancelar",
                                 width=12,
                                 command=self.cancelar)
        cancelar_but.grid(column=2, row=2)

        for frame in [self, frame_spins, frame_entry, frame_botones]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

    def min_spin_command(self, idx):
        val = self.min_vars[idx].get()
        if val > self.max_vars[idx].get():
            self.max_vars[idx].set(val)

    def max_spin_command(self, idx):
        val = self.max_vars[idx].get()
        if val < self.min_vars[idx].get():
            self.min_vars[idx].set(val)

    def set_min(self, idx):
        q_min = self.robot.q_min.tolist()
        new_q_i = float(self.min_vars[idx].get())
        q_min[idx] = new_q_i
        self.robot.q_min[idx] = new_q_i
        self.callback({'q_min': q_min})
        # self.robot.fkine(ones if check_tens else zeros[idx]==1)
        print(f'min q: {self.robot.q_min}')

    def set_max(self, idx):
        q_max = self.robot.q_max.tolist()
        new_q_i = float(self.max_vars[idx].get())
        q_max[idx] = new_q_i
        self.robot.q_max[idx] = new_q_i
        self.callback({'q_max': q_max})
        print(f'max q: {self.robot.q_max}')

    def jog(self):
        # self.robot.fkine(coprime_sines)
        pass

    def aceptar(self):
        # self.robot.fkine(zeros)
        self.callback({'headless' : bool(self.check_var.get()),
                       'p_scale' : float(self.p_scale_entry.get())})
        self.robot.stop_instance()
        self.destroy()

    def cancelar(self):
        # self.robot.fkine(zeros)
        self.callback(self.old_config)
        # Usar old config o limitar guardado a robot original?
        self.robot.stop_instance()
        self.destroy()