import tkinter as tk
from tkinter import ttk

from gui.menu_principal import PantallaMenuPrincipal
from gui.seleccion_robot import PantallaSelecRobot
from gui.seleccion_modelo import PantallaSelecModelo
from gui.config_ajuste import PantallaConfigAjuste
from gui.config_muestreo import PantallaConfigMuestreo
from gui.seleccion_puntos import PantallaSelecPuntos
from gui.resultados_pos import PantallaResultadosPosicion
from gui.progreso_ajuste import PantallaProgresoAjuste


class Interfaz(tk.Tk):

    def __init__(self):
        super().__init__()

        style= ttk.Style()
        style.configure('Red.TEntry', foreground='red')
        style.configure('Red.TButton', background='#FAA')
        style.map('Red.TButton', background=[('active', '#F66')])

        self.minsize(550,330)
        self.maxsize(1200,800)

        win_width = 800
        win_height = 450
        x_pos = int(self.winfo_screenwidth()/2 - win_width/2)
        y_pos = int(self.winfo_screenheight()/2 - win_height/2)
        geom = f'{win_width}x{win_height}+{x_pos}+{y_pos}'
        self.geometry(geom)

        # style= ttk.Style()
        # style.theme_use('clam')

        self.frame_stack = []
        self.ruta = []
        self.reset()

    def seleccionar_robot(self):
        self.ruta = [PantallaSelecRobot,
                     PantallaSelecModelo]
        self.avanzar()

    def controlar_robot(self):
        self.ruta = [PantallaSelecPuntos,
                     PantallaResultadosPosicion]
        self.avanzar()

    def entrenar_robot(self):
        self.ruta = [PantallaConfigAjuste,
                     PantallaConfigMuestreo,
                     PantallaProgresoAjuste]
        self.avanzar()

    def regresar(self):
        """
        Destruir el frame actual y enfocar el frame anterior en la ruta
        """
        self.frame_stack.pop().destroy()

    def avanzar(self):
        """
        Ir a la siguiente pantalla en la ruta actual
        """
        next_frame = self.ruta[len(self.frame_stack)-1]
        self.frame_stack.append(next_frame(self))

    def reset(self):
        for frame in self.frame_stack:
            frame.destroy()

        self.frame_stack = []
        self.ruta = []
        self.frame_stack.append(PantallaMenuPrincipal(self))


if __name__ == "__main__":
    
    gui = Interfaz()
    gui.mainloop()