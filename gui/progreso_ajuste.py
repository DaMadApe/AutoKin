import tkinter as tk
from tkinter import ttk
import webbrowser

from tensorboard import program


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


def abrir_tensorboard():
    pass
    webbrowser.open('')


if __name__ == '__main__':

    root = tk.Tk()
    root.minsize(550,330)
    root.maxsize(1200,800)

    win_width = 800
    win_height = 450
    x_pos = int(root.winfo_screenwidth()/2 - win_width/2)
    y_pos = int(root.winfo_screenheight()/2 - win_height/2)

    geom = f'{win_width}x{win_height}+{x_pos}+{y_pos}'
    root.geometry(geom)

    pant = PantallaProgresoAjuste(root)
    pant.abrir_tensorboard()
    #root.mainloop()