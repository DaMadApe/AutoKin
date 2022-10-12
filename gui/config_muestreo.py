from itertools import combinations

import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.gui_utils import Pantalla, Label_Entry
from autokin import trayectorias


class PantallaConfigMuestreo(Pantalla):

    samp_args = {
        'coprime_sines': {
            'n_points': {
                'label': '# de muestras',
                'var_type': 'int',
                'default_val': 100,
                'restr_positiv': True,
                'non_zero': True
            },
            'densidad': {
                'label': 'densidad de ondas',
                'var_type': 'int',
                'default_val': 1,
                'restr_positiv': True,
                'non_zero': False
            },
            'base_frec': {
                'label': 'frecuencia base',
                'var_type': 'int',
                'default_val': 1,
                'restr_positiv': True,
                'non_zero': False
            }
        }
    }

    def __init__(self, parent):
        self.arg_getters = None
        self.split_arg_getters = {}

        self.axis_combos = {}

        super().__init__(parent, titulo="Configurar muestreo")

    def definir_elementos(self):
        self.n_inputs = self.controlador.robot_s.n

        frame_grafica = ttk.Frame(self)
        frame_grafica.grid(column=0, row=0, sticky='nsew')

        # Opciones de trayectoria
        label_traj = ttk.Label(frame_grafica, 
                               text="Tipo de trayectoria")
        label_traj.grid(column=0, row=0, sticky='e')
        self.traj_combo = ttk.Combobox(frame_grafica, state='readonly')
        self.traj_combo.grid(column=1, row=0, sticky='w')
        self.traj_combo['values'] = list(self.samp_args.keys())
        self.traj_combo.bind('<<ComboboxSelected>>', self.definir_panel_config)
        self.traj_combo.set('coprime_sines')

        # Gráfica
        self.fig = Figure(figsize=(4,4), dpi=90)
        projection = '3d' if self.n_inputs > 2 else None
        self.ax = self.fig.add_subplot(projection=projection)
        self.fig.tight_layout()
        self.grafica = FigureCanvasTkAgg(self.fig, master=frame_grafica)
        self.grafica.get_tk_widget().grid(column=0, row=1, sticky='nsew',
                                          columnspan=2)

        # Selección de proyección en gráfica
        label_proy = ttk.Label(frame_grafica, 
                               text="Ver proyección sobre")
        label_proy.grid(column=0, row=2, sticky='e')
        self.proyec_combo = ttk.Combobox(frame_grafica, state='readonly', )
        self.proyec_combo.grid(column=1, row=2)

        if self.n_inputs <= 3:
            self.proyec_combo.config(state='disabled')
            proyecs = ['']
        else:
            combos = combinations(range(1, self.n_inputs+1),
                                        self.n_inputs-3)
            for i, combo in enumerate(combos):
                label = f"{i+1}."
                for ax in combo:
                    label += f" q{ax},"
                label = label[:-1]
                self.axis_combos[label] = combo
            proyecs = list(self.axis_combos.keys())

        self.proyec_combo['values'] = proyecs
        self.proyec_combo.set(proyecs[0])
        self.proyec_combo.bind('<<ComboboxSelected>>', self.recargar_grafica)

        # Configuraciones de la muestra
        frame_derecha = ttk.Frame(self)
        frame_derecha.grid(column=1, row=0, sticky='nsew')
        # Frame configs
        self.frame_configs = ttk.LabelFrame(frame_derecha, text="Parámetros")
        self.frame_configs.grid(column=0, row=0, sticky='nsew')
        self.definir_panel_config()

        # Frame config de split train-val-test
        self.frame_split = ttk.LabelFrame(frame_derecha, text="Reparto de datos")
        self.frame_split.grid(column=0, row=1, sticky='nsew')

        default_split = {'train': 0.7,
                         'val': 0.2,
                         'test': 0.1}
                    
        for i, label in enumerate(['train', 'val', 'test']):
            entry = Label_Entry(self.frame_split,
                                label=label,
                                var_type='float',
                                default_val=default_split[label],
                                restr_positiv=True)
            entry.grid(column=0, row=i)
            self.split_arg_getters[label] = entry.get

        # Botones
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.parent.regresar)
        boton_regresar.grid(column=0, row=1, sticky='w')

        boton_aceptar = ttk.Button(self, text="Ejecutar",
            command=self.ejecutar)
        boton_aceptar.grid(column=1, row=1, sticky='e')

        # Agregar pad a todos los widgets
        for frame in [frame_grafica, frame_derecha]:
            for child in frame.winfo_children():
                child.grid_configure(padx=10, pady=5)

        frame_grafica.grid_configure(padx=10, pady=5)
        frame_derecha.grid_configure(padx=10, pady=25)

        for child in self.frame_split.winfo_children():
            child.grid_configure(padx=10, pady=8)
 
        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        frame_grafica.rowconfigure(1, weight=1)
        frame_derecha.rowconfigure(0, weight=1)

    def definir_panel_config (self, *args):
        # Producir automáticamente entries según selección de trayectoria
        for widget in self.frame_configs.winfo_children():
            widget.destroy()

        self.arg_getters = {}

        args = self.samp_args[self.traj_combo.get()]

        for i, (arg_name, entry_kwargs) in enumerate(args.items()):
            entry = Label_Entry(self.frame_configs,
                                width=10, **entry_kwargs)
            entry.grid(column=0, row=i)
            entry.entry.bind('<Return>', self.recargar_grafica)
            entry.entry.bind('<FocusOut>', self.recargar_grafica)
            self.arg_getters[arg_name] = entry.get

        for child in self.frame_configs.winfo_children():
            child.grid_configure(padx=10, pady=8)

        self.recargar_grafica()

    def get_traj_kwargs(self):
        traj_kwargs = {}
        for arg_name, get_fn in self.arg_getters.items():
            traj_kwargs[arg_name] = get_fn()
        return traj_kwargs

    def get_split(self):
        split = {}
        for label, get_fn in self.split_arg_getters.items():
            val = get_fn()
            split[label] = val
            if val is None:
                return None
        return split

    def get_trayec(self):
        traj_cls = self.traj_combo.get()
        if traj_cls:
            traj_kwargs = self.get_traj_kwargs()
            if not (None in traj_kwargs.values()):
                traj_cls = getattr(trayectorias, traj_cls)
                trayec = traj_cls(self.n_inputs, **traj_kwargs)
                return trayec
        return None

    def get_proyec(self) -> list[int]:
        combo = self.proyec_combo.get()
        ejes_ocultos = self.axis_combos[combo] if combo else []
        return [i for i in range(self.n_inputs) if i+1 not in ejes_ocultos]

    def recargar_grafica(self, *args):
        self.ax.clear()
        trayec = self.get_trayec()
        ejes_visibles = self.get_proyec()
        if trayec is not None:
            puntosTranspuesto = trayec.transpose(0,1)# list(zip(*trayec))
            self.ax.plot(*puntosTranspuesto[ejes_visibles].numpy(),
                         color='lightcoral',
                         linewidth=1.5)
            self.ax.scatter(*puntosTranspuesto[ejes_visibles].numpy(),
                            color='red')
        self.ax.set_xlabel('q1')
        self.ax.set_ylabel('q2')
        if self.n_inputs > 2:
            self.ax.set_zlabel('q3')
        self.grafica.draw()

    def ejecutar(self):
        trayec = self.get_trayec()
        split = self.get_split()
        if trayec is not None and split is not None:
            self.controlador.set_sample(trayec, split)
            self.parent.avanzar()


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaConfigMuestreo(root)
    root.mainloop()