import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Pantalla


class PantallaResultadosPosicion(Pantalla):

    def __init__(self, parent):
        super().__init__(parent, titulo="Resultados en espacio de tareas")

    def definir_elementos(self):
        # Botones
        boton_aceptar = ttk.Button(self, text="Aceptar",
                                    command=self.parent.reset)
        boton_aceptar.grid(column=1, row=2, sticky='e')