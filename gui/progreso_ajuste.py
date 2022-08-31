import tkinter as tk
from tkinter import ttk
import webbrowser

from tensorboard import program

from gui.robot_database import UIController


class PantallaProgresoAjuste(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Progreso de entrenamiento")

        self.controlador = UIController()

        self.definir_elementos()

    def definir_elementos(self):

        # Botones
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.regresar)
        boton_regresar.grid(column=0, row=2, sticky='w')

        boton_continuar = ttk.Button(self, text="Aceptar",
                                     command=self.parent.reset)
        boton_continuar.grid(column=1, row=2, sticky='e')

    def regresar(self, *args):
        # TODO: Pedir confirmación, cancelar entrenamiento?
        self.parent.regresar()


def abrir_tensorboard():
    pass
    webbrowser.open('')


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaProgresoAjuste(root)
    #root.mainloop()