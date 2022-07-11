import tkinter as tk
from tkinter import ttk


class PantallaMenuPrincipal(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title("AutoKin")

        self.definir_elementos()

    def definir_elementos(self):
        pass