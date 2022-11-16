import tkinter as tk
from tkinter import ttk
import webbrowser

from tensorboard import program

from gui.gui_utils import Pantalla


class PantallaProgresoAjuste(Pantalla):

    def __init__(self, parent):
        self.status_labels = []
        self.etapa_actual = 0
        self.max_steps = 0
        self.terminado = False
        parent.after(100) # Solución extraña para evitar errores con
                          # callbacks cuando se rehace la pantalla
        super().__init__(parent, titulo="Progreso de entrenamiento")

    def definir_elementos(self):
        # Frame etapas + status
        frame_etapas = ttk.Frame(self)
        frame_etapas.grid(column=0, row=0)#, sticky='w')

        ttk.Label(frame_etapas, text="Etapas de ajuste", font=(13)).grid(column=0, row=0)

        self.etapas = list(self.controlador.train_kwargs.keys())
        if "Ajuste inicial" in self.etapas:
            self.etapas.insert(self.etapas.index("Ajuste inicial"),
                               "Muestreo inicial")

        for i, etapa in enumerate(self.etapas):
            ttk.Label(frame_etapas, text=etapa).grid(column=0, row=i+1, sticky='w')

            status_label = ttk.Label(frame_etapas, text="Pendiente")
            status_label.grid(column=1, row=i+1, sticky='w')
            status_label['style'] = 'Red.TLabel'

            self.status_labels.append(status_label)

        # Barra de progreso
        frame_progreso = ttk.Frame(self)
        frame_progreso.grid(column=0, row=1, columnspan=2, sticky='ew')
        frame_progreso.columnconfigure(0, weight=1)

        self.label_prog = ttk.Label(frame_progreso, text="Progreso: ")
        self.label_prog.grid(column=0, row=0, sticky='w')
        self.label_info = ttk.Label(frame_progreso, text=" ")
        self.label_info.grid(column=1, row=0, sticky='w')
        self.progreso = ttk.Progressbar(frame_progreso, orient='horizontal')
        self.progreso.grid(column=0, row=1, columnspan=2,
                           sticky='ew')
    
        boton_tb = ttk.Button(frame_progreso, text="Abrir Tensorboard",
                              command=self.abrir_tensorboard)
        boton_tb.grid(column=0, row=2, sticky='w')

        # Botones del fondo
        self.boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.regresar)
        self.boton_regresar.grid(column=0, row=3, sticky='sw')

        boton_continuar = ttk.Button(self, text="Salir",
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

    def _format_prog_info(self, progress_info):
        formated = str()
        for key, val in progress_info.items():
            formated += f"{key}: {val:4f}   "
        return formated

    def iniciar_entrenamiento(self):
        self.controlador.entrenar(self.stage_callback,
                                  self.step_callback,
                                  self.end_callback,
                                  self.parent.after)

    def stage_callback(self, steps: int):
        if str(self.progreso['mode']) == 'indeterminate':
            self.progreso.stop()
        else:
            self.progreso.step(1) # Último paso
        self.label_prog.config(text="")
        self.max_steps = steps
        if steps == 0:
            self.progreso.config(mode='indeterminate', maximum=100)
            self.progreso.start()
        else:
            self.progreso.config(mode='determinate', maximum=self.max_steps+1)
        if self.etapa_actual > 0:
            label = self.status_labels[self.etapa_actual-1]
            label.config(text="Completado")
            label['style'] = 'Green.TLabel'
        label = self.status_labels[self.etapa_actual]
        label.config(text="En proceso...")
        label['style'] = 'Orange.TLabel'

        self.etapa_actual += 1

    def step_callback(self, progress_info: dict, epoch: int):
        self.label_prog.config(text=f"Progreso: {epoch+1}/{self.max_steps}")
        self.label_info.config(text=self._format_prog_info(progress_info))
        self.progreso.step(1)
        self.update_idletasks()

    def end_callback(self):
        self.boton_regresar['state'] = 'disabled'
        if str(self.progreso['mode']) == 'indeterminate':
            self.progreso.stop()
        label = self.status_labels[-1]
        label.config(text="Completado")
        label['style'] = 'Green.TLabel'
        self.terminado = True

    def regresar(self, *args):
        self.controlador.pausar()
        if tk.messagebox.askokcancel("Regresar",
                                     "Cancelar entrenamiento?"):
            self.controlador.detener(guardar=False)
            # self.update_idletasks()
            self.parent.regresar()
        else:
            self.controlador.reanudar()

    def continuar(self, *args):
        if not self.terminado:
            self.controlador.pausar()
            if not tk.messagebox.askokcancel("Entrenamiento en progreso",
                                             "Cancelar entrenamiento?"):
                self.controlador.reanudar()
                return
        guardar_entrenamiento = tk.messagebox.askyesno("Guardar entrenamiento",
                                                       "Guardar progreso del modelo?")
        guardar_dataset = tk.messagebox.askyesno("Guardar dataset",
                                                 "Guardar datos muestreados?")
        self.controlador.detener(guardar_entrenamiento=guardar_entrenamiento,
                                 guardar_dataset=guardar_dataset)
        # self.update_idletasks()
        self.parent.reset()

    def abrir_tensorboard(self):
        self.controlador.abrir_tensorboard()

    def en_cierre(self):
        self.controlador.cerrar_tensorboard()


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