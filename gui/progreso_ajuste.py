import tkinter as tk
from tkinter import ttk


class PantallaProgresoAjuste(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Progreso de entrenamiento")

        self.definir_elementos()

    def definir_elementos(self):

        # Botones
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.parent.regresar)
        boton_regresar.grid(column=0, row=2, sticky='w')

        boton_continuar = ttk.Button(self, text="Aceptar",
                                    command=self.parent.reset)
        boton_continuar.grid(column=1, row=2, sticky='e')