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
                'default_val': 1000,
                'restr_positiv': True,
                'non_zero': True
            },
            'densidad': {
                'label': 'densidad de ondas',
                'var_type': 'int',
                'default_val': 0,
                'restr_positiv': True,
                'non_zero': False
            },
            'base_frec': {
                'label': 'frecuencia base',
                'var_type': 'int',
                'default_val': 0,
                'restr_positiv': True,
                'non_zero': False
            }
        }
    }

    def __init__(self, parent):
        self.arg_getters = None
        self.split_arg_getters = {}

        super().__init__(parent, titulo="Configurar muestreo")

    def definir_elementos(self):

        # Opciones de trayectoria
        self.traj_combo = ttk.Combobox(self, state='readonly')
        self.traj_combo.grid(column=0, row=0, sticky='w')
        self.traj_combo['values'] = list(self.samp_args.keys())
        self.traj_combo.bind('<<ComboboxSelected>>', self.definir_panel_config)
        self.traj_combo.set('coprime_sines')

        # Gráfica
        self.fig = Figure(figsize=(4,4), dpi=90)
        self.ax = self.fig.add_subplot(projection='3d')
        self.fig.tight_layout()
        self.grafica = FigureCanvasTkAgg(self.fig, master=self)
        self.grafica.get_tk_widget().grid(column=0, row=1, sticky='nw',
                                          rowspan=2, padx=5, pady=5)
        # self.recargar_grafica()

        # Frame configs
        self.frame_configs = ttk.LabelFrame(self, text='Parámetros')
        self.frame_configs.grid(column=1, row=1, sticky='nw')
        self.definir_panel_config()

        # Frame config de split train-val-test
        self.frame_split = ttk.LabelFrame(self, text='Reparto de datos')
        self.frame_split.grid(column=1, row=2, sticky='nw')

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
        boton_regresar.grid(column=0, row=3, sticky='w')

        boton_aceptar = ttk.Button(self, text="Ejecutar",
            command=self.ejecutar)
        boton_aceptar.grid(column=1, row=3, sticky='e')

        # Agregar pad a todos los widgets
        for frame in [self, self.frame_split]:
            for child in frame.winfo_children():
                child.grid_configure(padx=10, pady=5)
 
        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

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
            self.arg_getters[arg_name] = entry.get

        for child in self.frame_configs.winfo_children():
            child.grid_configure(padx=10, pady=5)

    def recargar_grafica(self):
        self.ax.clear()
        if self.puntos:
            puntosTranspuesto = list(zip(*self.puntos))
            self.ax.plot(*puntosTranspuesto[:3], color='lightcoral',
                        linewidth=1.5)
            self.ax.scatter(*puntosTranspuesto[:3], color='red')
        self.ax.set_xlabel('q1')
        self.ax.set_ylabel('q2')
        self.ax.set_zlabel('q3')
        self.grafica.draw()

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

    def ejecutar(self):
        n_inputs = self.controlador.robot_selec().robot.n

        traj_cls = self.traj_combo.get()

        if traj_cls:
            traj_kwargs = self.get_traj_kwargs()
            split = self.get_split()

            if not (None in traj_kwargs.values() or split is None):
                traj_cls = getattr(trayectorias, traj_cls)
                sample = traj_cls(n_inputs, **traj_kwargs)
                self.controlador.set_sample(sample, split)
                self.parent.avanzar()


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaConfigMuestreo(root)
    root.mainloop()