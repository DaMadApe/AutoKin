import tkinter as tk
from tkinter import ttk
import webbrowser

from tensorboard import program

from gui.gui_utils import Pantalla


class PantallaProgresoAjuste(Pantalla):

    def __init__(self, parent):
        self.status_labels = {}
        self.etapa_actual = 0
        super().__init__(parent, titulo="Progreso de entrenamiento")

    def definir_elementos(self):
        # Frame etapas + status
        frame_etapas = ttk.Frame(self)
        frame_etapas.grid(column=0, row=0)#, sticky='w')

        ttk.Label(frame_etapas, text="Etapas de ajuste", font=(13)).grid(column=0, row=0)

        self.etapas = list(self.controlador.train_kwargs.keys())
        if "Ajuste inicial" in self.etapas:
            self.etapas.insert(self.etapas.index("Ajuste inicial")-1,
                          "Muestreo inicial")

        for i, etapa in enumerate(self.etapas):
            ttk.Label(frame_etapas, text=etapa).grid(column=0, row=i+1, sticky='w')

            status_label = ttk.Label(frame_etapas, text="Pendiente")
            status_label.grid(column=1, row=i+1, sticky='w')
            status_label['style'] = 'Red.TLabel'

            self.status_labels[etapa] = status_label

        # Barra de progreso
        frame_progreso = ttk.Frame(self)
        frame_progreso.grid(column=0, row=1, columnspan=2, sticky='ew')
        frame_progreso.columnconfigure(0, weight=1)

        self.label_prog = ttk.Label(frame_progreso, text="Progreso: ")
        self.label_prog.grid(column=0, row=0, sticky='w')
        self.label_info = ttk.Label(frame_progreso, text=" ")
        self.label_info.grid(column=1, row=0, sticky='w')
        self.progreso = ttk.Progressbar(frame_progreso, orient='horizontal',
                                        mode='determinate')
        self.progreso.grid(column=0, row=1, columnspan=2,
                           sticky='ew')
    
        boton_tb = ttk.Button(frame_progreso, text="Abrir Tensorboard",
                              command=self.abrir_tensorboard)
        boton_tb.grid(column=0, row=2, sticky='w')

        # Botones del fondo
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.regresar)
        boton_regresar.grid(column=0, row=3, sticky='sw')

        boton_continuar = ttk.Button(self, text="Aceptar",
                                     command=self.continuar)
        boton_continuar.grid(column=1, row=3, sticky='se')

        # Adaptar tamaño de componentes
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=2)

        # Agregar pad a todos los widgets
        for frame in [self, frame_etapas, frame_progreso]:
            for child in frame.winfo_children():
                child.grid_configure(padx=10, pady=5)

        self.parent.after(100, self.iniciar_entrenamiento)

    def iniciar_entrenamiento(self):
        epochs = self.controlador.train_kwargs['Ajuste inicial']['epochs']
        self.progreso.config(maximum=epochs)
        def step_callback(progress_info, epoch):
            self.label_prog.config(text=f"Progreso: {epoch}")
            self.label_info.config(text=self._format_prog_info(progress_info))
            self.parent.after(50, self.progreso.step)
        def close_callback():
            self.progreso.stop()
        def stage_callback():
            label = self.status_labels[self.etapas[self.etapa_actual]]
            label.config(text="Completado")
            label['style'] = 'Green.TLabel'
            self.etapa_actual += 1
        self.controlador.entrenar(stage_callback, step_callback, close_callback)

    def _format_prog_info(self, progress_info):
        formated = str()
        for key, val in progress_info.items():
            formated += f"{key}: {val:4f}   "
        return formated

    def regresar(self, *args):
        if tk.messagebox.askokcancel("Cerrar", "Cerrar programa?"):
            self.parent.regresar()
            self.progreso.stop()

    def continuar(self, *args):
        # self.progreso.step()
        self.parent.reset()

    def abrir_tensorboard(self):
        nom_robot = self.controlador.robot_selec.nombre
        nom_modelo = self.controlador.modelo_selec.nombre
        log_path = f'{self.controlador.tb_dir}/{nom_robot}_{nom_modelo}'

        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', log_path])
        url = self.tb.launch()

        webbrowser.open(url)

        # TODO: Proceso de tensorboard se queda abierto, buscar
        #       forma de detener o reemplazar nuevas instancias


if __name__ == '__main__':
    from gui.gui_utils import MockInterfaz
    from gui.gui_control import UIController

    root = MockInterfaz()

    train_kwargs = {'Meta ajuste': {'epochs': 1000},
                    'Ajuste inicial': {'epochs': 1000, 'lr': 0.001, 'batch_size': 256},
                    'Ajuste dirigido': {'query_steps': 5, 'n_queries': 5}}

    UIController().set_train_kwargs(train_kwargs)

    PantallaProgresoAjuste(root)

    root.mainloop()