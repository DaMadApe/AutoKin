import time
import logging
from functools import partial
import tkinter as tk
from tkinter import ttk

import torch

from autokin.robot import ExternRobot, RTBrobot, SofaRobot
from autokin.trayectorias import coprime_sines
from autokin.utils import RobotExecError, restringir
from gui.gui_utils import Popup, Label_Entry


def pos_scale_offset(p_sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    max_p = p_sample.max(dim=0).values
    min_p = p_sample.min(dim=0).values
    logging.info(f"max_p: {max_p}, min_p: {min_p}")

    p_scale = 1/(max_p - min_p)
    p_offset = -min_p*p_scale

    return p_scale, p_offset


class Popup_config_ext(Popup):

    def __init__(self, parent, callback, robot: ExternRobot):
        self.callback = callback
        self.robot = robot
        self.min_vars = []
        self.max_vars = []
        self.spins = []

        self.old_config = {'q_min': robot.q_min,
                           'q_max': robot.q_max}
        super().__init__(title="Configurar robot: Externo", parent=parent)

    def definir_elementos(self):
        # Configuración de mínimos/máximos de actuación
        frame_spins = ttk.Frame(self)
        frame_spins.grid(column=0, row=0)

        self.protocol("WM_DELETE_WINDOW", self.cancelar)

        for i in range(self.robot.n):
            q_label = ttk.Label(frame_spins, text=f"q{i+1}")
            q_label.grid(column=0, row=i)

            min_var = tk.IntVar(value=int(self.robot.q_min[i]))
            self.min_vars.append(min_var)
            max_var = tk.IntVar(value=int(self.robot.q_max[i]))
            self.max_vars.append(max_var)

            min_spin = ttk.Spinbox(frame_spins, width=5,
                                   from_=0, to=1000, increment=1,
                                   textvariable=min_var, 
                                   command=partial(self.min_spin_command, i))
            min_spin.grid(column=1, row=i)

            max_spin = ttk.Spinbox(frame_spins, width=5,
                                   from_=0, to=1000, increment=1,
                                   textvariable=max_var, 
                                   command=partial(self.max_spin_command, i))
            max_spin.grid(column=3, row=i)

            self.spins.extend([min_spin, max_spin])

            min_but = ttk.Button(frame_spins, text="Set min",
                                 width=7,
                                 command=partial(self.set_min, i))
            min_but.grid(column=2, row=i)

            max_but = ttk.Button(frame_spins, text="Set max",
                                 width=7,
                                 command=partial(self.set_max, i))
            max_but.grid(column=4, row=i)

        # Selector de tamaño de paso
        frame_step_size = ttk.LabelFrame(self, text="Tamaño de paso")
        frame_step_size.grid(column=0, row=1, sticky='ew')

        # Tamaños disponibles de paso
        # self.step_sizes = [1, 10, 100]
    
        self.step_sel = tk.IntVar(value=0)
        for i in range(3):
            step_radio = ttk.Radiobutton(frame_step_size,
                                        text=f'x {10**i}', # f'x{step_sizes[i]}',
                                        variable=self.step_sel,
                                        value=i,
                                        command=partial(self.set_step_size, i))
            step_radio.grid(column=i, row=0, sticky='ew', pady=0, padx=20)
            frame_step_size.columnconfigure(i, weight=1)

        # Definir factor de escala para medidas de posición
        frame_dq = ttk.Frame(self)
        frame_dq.grid(column=0, row=2, sticky='w')

        # Configurar máximo cambio instantáneo de actuadores
        max_dq_label = ttk.Label(frame_dq, text=f"Máximo dq instantáneo")
        max_dq_label.grid(column=0, row=0)

        self.max_dq_var = tk.DoubleVar(value=self.robot.max_dq)
        max_dq_spin = ttk.Spinbox(frame_dq, width=5,
                                  from_=1, to=30, increment=1,
                                  textvariable=self.max_dq_var)
        max_dq_spin.grid(column=1, row=0)

        set_dq_but = ttk.Button(frame_dq, text="Set",
                                width=4,
                                command=self.set_max_dq)
        set_dq_but.grid(column=2, row=0)

        # Botón recalibrar p
        but_recal_p = ttk.Button(self, text="Recalibrar escala de posiciones",
                                width=42,
                                command=self.recal_p)
        but_recal_p.grid(column=0, row=3)
 
        # Botones del fondo
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=0, row=4)

        jog_but = ttk.Button(frame_botones, text="Jog",
                             width=12,
                             command=self.jog)
        jog_but.grid(column=0, row=2)

        aceptar_but = ttk.Button(frame_botones, text="Aceptar",
                                 width=12,
                                 command=self.aceptar)
        aceptar_but.grid(column=1, row=2)

        cancelar_but = ttk.Button(frame_botones, text="Cancelar",
                                 width=12,
                                 command=self.cancelar)
        cancelar_but.grid(column=2, row=2)

        for frame in [self, frame_spins, frame_dq, frame_botones]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

        frame_step_size.grid_configure(padx=10, pady=5)

    def _fkine(self, q):
        try:
            q, p = self.robot.fkine(q)
        except RobotExecError:
            tk.messagebox.showerror("Robot error",
                                    "Error de ejecución en el robot")
            return None
        else:
            return q, p

    def min_spin_command(self, idx): 
        val = self.min_vars[idx].get()
        if val > self.max_vars[idx].get():
            self.max_vars[idx].set(val)

    def max_spin_command(self, idx):
        val = self.max_vars[idx].get()
        if val < self.min_vars[idx].get():
            self.min_vars[idx].set(val)

    def set_step_size(self, idx):
        for spin in self.spins:
            spin.config(increment=10**idx)

    def set_min(self, idx):
        q_min = self.robot.q_min
        q_min[idx] = float(self.min_vars[idx].get())
        self.callback({'q_min': q_min})

        self._fkine(torch.zeros(self.robot.n))

    def set_max(self, idx):
        q_max = self.robot.q_max
        q_max[idx] = float(self.max_vars[idx].get())
        self.callback({'q_max': q_max})

        q = torch.zeros(self.robot.n)
        q[idx] = 1
        self._fkine(q)

    def set_max_dq(self):
        max_dq = float(self.max_dq_var.get())
        self.callback({'max_dq': max_dq})

    def recal_p(self):
        msg = """Recalibrar escala de posiciones? Puede invalidar modelos previamente entrenados."""
        if tk.messagebox.askokcancel("Recalibrar norma de p", msg):
            zero = torch.zeros(1, self.robot.n)
            jog_traj = restringir(coprime_sines(self.robot.n, 300, densidad=0))
            traj = torch.concat([zero, jog_traj, zero])

            sample = self._fkine(traj)
            if sample is not None:
                p_scale, p_offset = pos_scale_offset(sample)
                self.callback({
                    'p_scale' : p_scale,
                    'p_offset' : p_offset
                })

    def jog(self):
        zero = torch.zeros(1, self.robot.n)
        jog_traj = restringir(coprime_sines(self.robot.n, 1000, densidad=0))
        traj = torch.concat([zero, jog_traj, zero])

        self._fkine(traj)

    def aceptar(self):
        q_min_list = [var.get() for var in self.min_vars]
        q_max_list = [var.get() for var in self.max_vars]
        self.callback({'q_min': torch.tensor(q_min_list),
                       'q_max': torch.tensor(q_max_list),
                       'max_dq': float(self.max_dq_var.get())})
        self._fkine(torch.zeros(self.robot.n))
        self.destroy()

    def cancelar(self):
        self.callback(self.old_config)
        self._fkine(torch.zeros(self.robot.n))
        self.destroy()


class Popup_config_rtb(Popup):
    def __init__(self, parent, callback, robot: RTBrobot):
        pass
    #     self.callback = callback
    #     self.robot = robot
    #     super().__init__(title="Configurar robot: RTB", parent=parent)

    # def definir_elementos(self):
        # pass


class Popup_config_sofa(Popup):

    def __init__(self, parent, callback, robot: SofaRobot):
        self.callback = callback
        self.robot = robot
        self.min_vars = []
        self.max_vars = []

        self.old_config = {'q_min': robot.q_min,
                           'q_max': robot.q_max,
                           'headless': robot.headless}
        super().__init__(title="Configurar robot: Sofa", parent=parent)

        self.robot.headless = False
        self.robot.start_instance()
        time.sleep(2)
        self.lift()

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
                                   from_=0.0, to=30.0, increment=0.5,
                                   textvariable=min_var, 
                                   command=partial(self.min_spin_command, i))
            min_spin.grid(column=1, row=i)

            max_spin = ttk.Spinbox(frame_spins, width=5,
                                   from_=0.0, to=30.0, increment=0.5,
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
        frame_dq = ttk.Frame(self)
        frame_dq.grid(column=0, row=1, sticky='w')

        # Configurar máximo cambio instantáneo de actuadores
        max_dq_label = ttk.Label(frame_dq, text=f"Máximo dq instantáneo")
        max_dq_label.grid(column=0, row=0)

        self.max_dq_var = tk.DoubleVar(value=float(self.robot.max_dq))
        max_dq_spin = ttk.Spinbox(frame_dq, width=5,
                                  from_=0.01, to=50.0, increment=0.5,
                                  textvariable=self.max_dq_var)
        max_dq_spin.grid(column=1, row=0)

        set_dq_but = ttk.Button(frame_dq, text="Set",
                                width=4,
                                command=self.set_max_dq)
        set_dq_but.grid(column=2, row=0)
 
        # Botón recalibrar p
        but_recal_p = ttk.Button(self, text="Recalibrar escala de posiciones",
                                width=42,
                                command=self.recal_p)
        but_recal_p.grid(column=0, row=2)

        # Botones del fondo
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=0, row=3)

        self.hdls_check = tk.IntVar(value=self.robot.headless)
        check_hdls = ttk.Checkbutton(frame_botones,
                                     text='Usar interfaz gráfica',
                                     variable=self.hdls_check)
        check_hdls.grid(column=0, row=1, columnspan=3, sticky='w')

        jog_but = ttk.Button(frame_botones, text="Jog",
                             width=12,
                             command=self.jog)
        jog_but.grid(column=0, row=2)

        aceptar_but = ttk.Button(frame_botones, text="Aceptar",
                                 width=12,
                                 command=self.aceptar)
        aceptar_but.grid(column=1, row=2)

        cancelar_but = ttk.Button(frame_botones, text="Cancelar",
                                 width=12,
                                 command=self.cancelar)
        cancelar_but.grid(column=2, row=2)

        for frame in [self, frame_spins, frame_dq, frame_botones]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

    def _fkine(self, q):
        if self.robot.running():
            try:
                q, p = self.robot.fkine(q)
            except RobotExecError:
                tk.messagebox.showerror("Robot error",
                                        "Error de ejecución en el robot")
                return None
            else:
                return q, p

    def min_spin_command(self, idx): 
        val = self.min_vars[idx].get()
        if val > self.max_vars[idx].get():
            self.max_vars[idx].set(val)

    def max_spin_command(self, idx):
        val = self.max_vars[idx].get()
        if val < self.min_vars[idx].get():
            self.min_vars[idx].set(val)

    def set_min(self, idx):
        q_min = self.robot.q_min
        q_min[idx] = float(self.min_vars[idx].get())
        self.callback({'q_min': q_min})

        self._fkine(torch.zeros(self.robot.n))

    def set_max(self, idx):
        q_max = self.robot.q_max
        q_max[idx] = float(self.max_vars[idx].get())
        self.callback({'q_max': q_max})

        q = torch.zeros(self.robot.n)
        q[idx] = 1
        self._fkine(q)

    def set_max_dq(self):
        max_dq = float(self.max_dq_var.get())
        self.callback({'max_dq': max_dq})

    def recal_p(self):
        msg = """Recalibrar escala de posiciones? Puede invalidar modelos previamente entrenados."""
        if tk.messagebox.askokcancel("Recalibrar norma de p", msg):
            zero = torch.zeros(1, self.robot.n)
            jog_traj = restringir(coprime_sines(self.robot.n, 300, densidad=0))
            traj = torch.concat([zero, jog_traj, zero])

            sample = self._fkine(traj)
            if sample is not None:
                p_scale, p_offset = pos_scale_offset(sample)
                self.callback({
                    'p_scale' : p_scale,
                    'p_offset' : p_offset
                })

    def jog(self):
        zero = torch.zeros(1, self.robot.n)
        jog_traj = restringir(coprime_sines(self.robot.n, 1000, densidad=0))
        traj = torch.concat([zero, jog_traj, zero])

        self._fkine(traj)

    def aceptar(self):
        q_min_list = [var.get() for var in self.min_vars]
        q_max_list = [var.get() for var in self.max_vars]
        self.callback({
            'q_min': torch.tensor(q_min_list),
            'q_max': torch.tensor(q_max_list),
            'max_dq': float(self.max_dq_var.get()),
            'headless': bool(self.hdls_check.get())})
        self.robot.stop_instance()
        self.destroy()

    def cancelar(self):
        self.callback(self.old_config)
        self.robot.stop_instance()
        self.destroy()