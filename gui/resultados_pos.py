import tkinter as tk
from tkinter import ttk


class PantallaResultadosPosicion(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Resultados en espacio de tareas")

        self.definir_elementos()

    def definir_elementos(self):

        # Botones
        boton_aceptar = ttk.Button(self, text="Aceptar",
                                    command=self.parent.reset)
        boton_aceptar.grid(column=1, row=2, sticky='e')