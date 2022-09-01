import tkinter as tk
from tkinter import ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.robot_database import UIController


class PantallaConfigMuestreo(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Configurar muestreo")

        self.controlador = UIController()

        self.definir_elementos()

    def definir_elementos(self):

        # Frame lista + gráfica + opciones de proyección
        frame_grafica = ttk.Frame(self)
        frame_grafica.grid(column=0, row=0)

        # Gráfica
        # self.fig = Figure(figsize=(8,8), dpi=90)
        # self.ax = self.fig.add_subplot(projection='3d')
        # self.fig.tight_layout()
        # self.grafica = FigureCanvasTkAgg(self.fig, master=frame_grafica)
        # self.grafica.get_tk_widget().grid(column=1, row=0, sticky='n',
        #                                   rowspan=3, padx=5, pady=5)
        # self.recargar_grafica()

        # Frame configs

        # Agregar config de split train-val-test

        # Botones
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.parent.regresar)
        boton_regresar.grid(column=0, row=2, sticky='w')

        boton_aceptar = ttk.Button(self, text="Aceptar",
            command=self.parent.avanzar)
        boton_aceptar.grid(column=1, row=2, sticky='e')


    def recargar_grafica(self):
        self.ax.clear()
        if self.puntos:
            puntosTranspuesto = list(zip(*self.puntos))
            self.ax.plot(*puntosTranspuesto[:3], color='lightcoral',
                        linewidth=1.5)
            self.ax.scatter(*puntosTranspuesto[:3], color='red')
        self.ax.set_xlabel('q1')
        self.ax.set_ylabel('q2')
        self.ax.set_zlabel('q3')
        self.grafica.draw()