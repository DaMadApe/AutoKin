import tkinter as tk

from menu_principal import PantallaMenuPrincipal
from seleccion_robot import PantallaSelecRobot
from config_ajuste import PantallaConfigAjuste
from config_muestreo import PantallaConfigMuestreo
from seleccion_puntos import PantallaSelecPuntos
from resultados_pos import PantallaResultadosPosicion
from progreso_ajuste import PantallaProgresoAjuste


class Interfaz(tk.Tk):

    def __init__(self):
        super().__init__()

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

        pant5 = PantallaSelecPuntos(self)
        self.mainloop()