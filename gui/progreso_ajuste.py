import tkinter as tk
from tkinter import ttk
import webbrowser

from tensorboard import program

from gui.gui_utils import Pantalla


class PantallaProgresoAjuste(Pantalla):

    def __init__(self, parent):
        super().__init__(parent, titulo="Progreso de entrenamiento")

    def definir_elementos(self):

        train_kwargs = self.controlador.train_kwargs

        # Frame etapas + status
        frame_etapas = ttk.Frame(self)
        frame_etapas.grid(column=0, row=0)#, sticky='w')

        ttk.Label(frame_etapas, text="Etapas de ajuste", font=(13)).grid(column=0, row=0)

        for i, etapa in enumerate(self.controlador.train_kwargs.keys()):
            ttk.Label(frame_etapas, text=etapa).grid(column=0, row=i+1, sticky='w')

            status_label = ttk.Label(frame_etapas, text="Pendiente")
            status_label.grid(column=1, row=i+1, sticky='w')
            status_label['style'] = 'Red.TLabel'

        # Barra de progreso
        frame_progreso = ttk.Frame(self)
        frame_progreso.grid(column=0, row=1, columnspan=2, sticky='ew')
        frame_progreso.columnconfigure(0, weight=1)

        self.label_prog = ttk.Label(frame_progreso, text="Progreso: ")
        self.label_prog.grid(column=0, row=0, sticky='w')
        self.progreso = ttk.Progressbar(frame_progreso, orient='horizontal',
                                        mode='indeterminate')
        self.progreso.grid(column=0, row=1, sticky='ew')
    
        boton_tb = ttk.Button(frame_progreso, text="Abrir Tensorboard",
                              command=self.abrir_tensorboard)
        boton_tb.grid(column=0, row=2, sticky='e')

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

        self.iniciar_entrenamiento()

    def iniciar_entrenamiento(self):
        self.progreso.start()
        self.controlador.entrenar()

    def regresar(self, *args):
        # TODO: Pedir confirmación, cancelar entrenamiento?
        # self.parent.regresar()
        self.progreso.stop()

    def continuar(self, *args):
        # self.progreso.step()
        self.parent.reset()

    def abrir_tensorboard(self):
        nom_robot = self.controlador.robot_selec().nombre
        nom_modelo = self.controlador.modelo_selec().nombre
        log_path = f'gui/app_data/tb_logs/{nom_robot}_{nom_modelo}'

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