import tkinter as tk
from tkinter import ttk


class PantallaProgresoAjuste(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title("Progreso de entrenamiento")

        self.definir_elementos()

    def definir_elementos(self):
        pass