import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.gui_utils import Pantalla, TablaYBotones


class PantallaResultadosPosicion(Pantalla):

    def __init__(self, parent):
        self.medidas = []

        super().__init__(parent, titulo="Resultados en espacio de tareas")

    def definir_elementos(self):
        self.objetivos = self.controlador.puntos

        # Gráfica
        self.fig = Figure(figsize=(8,8), dpi=90)
        self.ax = self.fig.add_subplot(projection='3d')
        self.fig.tight_layout()
        self.grafica = FigureCanvasTkAgg(self.fig, master=self)
        self.grafica.get_tk_widget().grid(column=0, row=0, sticky='nw')
        self.recargar_grafica()

        # Tabla de puntos
        columnas = ('i', 'x*', 'y*', 'z*', 'x_r', 'y_r', 'z_r', 'e')
        self.tabla = TablaYBotones(self, botones_abajo=True,
                                   columnas=columnas,
                                   anchos=(30, 50, 50, 50,
                                           50, 50, 50, 50))
        self.tabla.grid(column=1, row=0, rowspan=2, sticky='nse')

        # Check si mostrar objetivos
        check_var = tk.IntVar(value=1)
        self.check_objs = ttk.Checkbutton(self,
                                          text="Mostrar objetivos",
                                          variable=check_var,
                                          command=self.recargar_grafica)
        self.check_objs .grid(column=0, row=1, sticky='w')

        # Botones
        boton_aceptar = ttk.Button(self, text="Aceptar",
                                    command=self.parent.reset)
        boton_aceptar.grid(column=1, row=2, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
    
        self.parent.after(100, self.iniciar)

    def _errorL2(self, x1, y1, z1, x2, y2, z2):
        return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**0.5

    def _truncar_punto(self, punto):
        return tuple((round(x, ndigits=3) for x in punto))

    def reg_callback(self, punto):
        self.parent.after(100)
        self.medidas.append([*punto])
        print(self.medidas)
        self.recargar_tabla()
        self.recargar_grafica()
        self.parent.after(100)

    def iniciar(self):
        self.controlador.ejecutar_trayec(self.reg_callback)

    def recargar_tabla(self):
        self.tabla.limpiar_tabla()
        for i, objetivo in enumerate(self.objetivos):
            obj_trunco = self._truncar_punto(objetivo[:3])
            if i < len(self.medidas):
                med_trunca = self._truncar_punto(self.medidas[i])
                error = self._errorL2(*obj_trunco, *med_trunca)
                entrada = (i, *obj_trunco, *med_trunca, error)
            else:
                entrada = (i, *obj_trunco)
            self.tabla.agregar_entrada(*entrada)

        self.tabla.desactivar_botones()

    def recargar_grafica(self):
        self.ax.clear()
        objs_transpuesto = list(zip(*self.objetivos))
        self.ax.plot(*objs_transpuesto[:3], color='lightcoral',
                     linewidth=1.5)
        self.ax.scatter(*objs_transpuesto[:3], color='red')

        if self.medidas:
            meds_transpuesto = list(zip(*self.medidas))
            self.ax.plot(*meds_transpuesto, color='orange',
                         linewidth=1.5)
            self.ax.scatter(*meds_transpuesto, color='darkorange')
            print(meds_transpuesto)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.grafica.draw()


if __name__ == '__main__':
    import torch

    from autokin.trayectorias import coprime_sines
    from gui.gui_utils import MockInterfaz
    from gui.gui_control import UIController

    root = MockInterfaz()

    controlador = UIController()

    q_demo = coprime_sines(controlador.robot_selec.robot.n, 80)[:10]
    _, p_demo = controlador.robot_selec.robot.fkine(q_demo)
    puntos = torch.concat([p_demo,
                           torch.zeros((len(p_demo),1)),
                           0.1*torch.ones((len(p_demo),1))], dim=1)
    puntos = puntos.tolist()

    controlador.set_trayec(puntos)

    PantallaResultadosPosicion(root)

    root.mainloop()