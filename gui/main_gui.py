import tkinter as tk

from gui.menu_principal import PantallaMenuPrincipal
from gui.seleccion_robot import PantallaSelecRobot
from gui.config_ajuste import PantallaConfigAjuste
from gui.config_muestreo import PantallaConfigMuestreo
from gui.seleccion_puntos import PantallaSelecPuntos
from gui.resultados_pos import PantallaResultadosPosicion
from gui.progreso_ajuste import PantallaProgresoAjuste

from autokin import modelos


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

        self.frame_stack = []
        self.ruta = []
        self.reset()

    def seleccionar_robot(self):
        self.frame_stack.append(PantallaSelecRobot(self))

    def controlar_robot(self):
        self.ruta = [PantallaSelecPuntos,
                     PantallaResultadosPosicion]
        self.frame_stack.append(PantallaSelecPuntos(self))

    def entrenar_robot(self):
        self.ruta = [PantallaConfigAjuste,
                     PantallaConfigMuestreo,
                     PantallaProgresoAjuste]
        self.frame_stack.append(PantallaConfigAjuste(self))

    def set_model(self, model_cls, model_kwargs, train_kwargs):
        model = model_cls(input_dim=self.selected_robot.n,
                          output_dim=3, **model_kwargs)
        # TODO: Rehacer sin pseudocódigo
        self.configs['train_kwargs'] = train_kwargs

    def avanzar(self, caller):
        idx = self.ruta.index(caller)
        self.frame_stack.append(self.ruta[idx+1](self))

    def reset(self):
        for frame in self.frame_stack:
            frame.destroy()

        self.frame_stack = []
        self.ruta = []
        self.frame_stack.append(PantallaMenuPrincipal(self))


if __name__ == "__main__":
    
    gui = Interfaz()
    gui.mainloop()