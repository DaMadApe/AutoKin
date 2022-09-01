import tkinter as tk
from tkinter import ttk
import webbrowser

from tensorboard import program

from gui.robot_database import UIController


class PantallaProgresoAjuste(ttk.Frame):

    def __init__(self, parent, train_kwargs):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Progreso de entrenamiento")

        self.controlador = UIController()

        self.train_kwargs = train_kwargs

        self.definir_elementos()

    def definir_elementos(self):

        # Frame etapas + status
        frame_etapas = ttk.Frame(self)
        frame_etapas.grid(column=0, row=0)#, sticky='w')

        ttk.Label(frame_etapas, text="Etapas de ajuste", font=(13)).grid(column=0, row=0)

        for i, etapa in enumerate(self.train_kwargs.keys()):# controlador.get_train_kwargs().keys():
            ttk.Label(frame_etapas, text=etapa).grid(column=0, row=i+1, sticky='w')

            status_label = ttk.Label(frame_etapas, text="Pendiente")
            status_label.grid(column=1, row=i+1, sticky='w')
            status_label['style'] = 'Red.TLabel'

        # Barra de progreso
        frame_progreso = ttk.Frame(self)
        frame_progreso.grid(column=0, row=1, columnspan=2, sticky='ew')
        frame_progreso.columnconfigure(0, weight=1)

        self.label_prog = ttk.Label(frame_progreso, text="Progreso: ")
        self.label_prog.grid(column=0, row=0)#, pady=10)
        self.progreso = ttk.Progressbar(frame_progreso, orient='horizontal',
                                        mode='indeterminate')
        self.progreso.grid(column=0, row=1, sticky='ew')#, padx=10, pady=5)
    
        boton_tb = ttk.Button(frame_progreso, text="Abrir Tensorboard",
                              command=self.abrir_tensorboard)
        boton_tb.grid(column=0, row=2, sticky='e')

        # Botones del fondo
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.regresar)
        boton_regresar.grid(column=0, row=3, sticky='sw')

        boton_continuar = ttk.Button(self, text="Aceptar",
                                     command=self.paso_progreso) #parent.reset)
        boton_continuar.grid(column=1, row=3, sticky='se')

        # Adaptar tamaño de componentes
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=2)

        # Agregar pad a todos los widgets
        for frame in [self, frame_etapas, frame_progreso]:
            for child in frame.winfo_children():
                child.grid_configure(padx=10, pady=5)

    def regresar(self, *args):
        # TODO: Pedir confirmación, cancelar entrenamiento?
        # self.parent.regresar()
        self.progreso.stop()

    def paso_progreso(self, *args):
        self.progreso.start()


    def abrir_tensorboard(self):
        log_path = 'autokin/experimentos/tb_logs/p_ajuste'

        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', log_path])
        url = self.tb.launch()

        webbrowser.open(url)

        # TODO: Proceso de tensorboard se queda abierto, buscar
        #       forma de detener o reemplazar nuevas instancias


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()


    train_kwargs = {'Meta ajuste': {'epochs': 1000}, 'Ajuste inicial': {'epochs': 1000, 'lr': 0.001, 'batch_size': 256}, 'Ajuste dirigido': {'query_steps': 5, 'n_queries': 5}}

    PantallaProgresoAjuste(root, train_kwargs)

    root.mainloop()