import os

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.gui_utils import Pantalla, Label_Entry, TablaYBotones


class PantallaSelecPuntos(Pantalla):

    def __init__(self, parent):
        self.puntos = []

        super().__init__(parent, titulo="Selección de puntos")

    def definir_elementos(self):
        columnas = ('i', 'x', 'y', 'z', 'tt', 'ts')
        self.tabla = TablaYBotones(self, botones_abajo=True,
                                   columnas=columnas,
                                   anchos=(30, 50, 50, 50, 50, 50),
                                   fn_doble_click=self.agregar_punto)
        self.tabla.grid(column=0, row=0, sticky='nsew')

        self.tabla.agregar_boton(text="Agregar punto",
                                 command=self.agregar_punto,
                                 padx=(0,5))

        self.tabla.agregar_boton(text="Borrar punto",
                                 command=self.borrar_punto,
                                 activo_en_seleccion=True,
                                 padx=(10,15))

        self.tabla.agregar_boton(text="Limpiar",
                                 command=self.limpiar,
                                 rojo=True,
                                 padx=(5,0))

        # Guardar/Cargar lista de puntos
        frame_guardar = ttk.Frame(self)
        frame_guardar.grid(column=0, row=1, sticky='s')

        boton_guardar = ttk.Button(frame_guardar, text="Guardar",
                                   command=self.guardar_puntos)
        boton_guardar.grid(column=0, row=0, padx=(0, 5))

        boton_cargar = ttk.Button(frame_guardar, text="Cargar",
                                  command=self.cargar_puntos)
        boton_cargar.grid(column=1, row=0, padx=(0, 5))

        self.listas = ttk.Combobox(frame_guardar) 
        self.listas.grid(column=2, row=0, padx=(5, 0))
        self.cargar_listas()

        # Gráfica
        self.fig = Figure(figsize=(8,8), dpi=90)
        self.ax = self.fig.add_subplot(projection='3d')
        self.fig.tight_layout()
        self.grafica = FigureCanvasTkAgg(self.fig, master=self)
        self.grafica.get_tk_widget().grid(column=1, row=0, sticky='n',
                                          rowspan=3, padx=5, pady=5)
        self.recargar_grafica()

        # Configuraciones del movimiento
        frame_configs = ttk.LabelFrame(self)
        frame_configs.grid(column=0, row=2, sticky='nsew')

        self.config1_entry = Label_Entry(frame_configs,
                                         label="Config 1",
                                         var_type='float',
                                         width=10)
        self.config1_entry.grid(column=0, row=0)

        self.config2_entry = Label_Entry(frame_configs,
                                         label="Config 2",
                                         var_type='float',
                                         width=10)
        self.config2_entry.grid(column=0, row=1)

        # Botones
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.parent.regresar)
        boton_regresar.grid(column=0, row=3, sticky='w')

        boton_ejecutar = ttk.Button(self, text="Ejecutar",
            command=self.parent.avanzar)
        boton_ejecutar.grid(column=1, row=3, sticky='e')

        # Agregar pad a todos los widgets
        for frame in [self, frame_configs]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=6)

        # Comportamiento al cambiar de tamaño
        # self.tabla.rowconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    def agregar_punto(self, indice):
        def callback(punto):
            self.puntos.insert(indice, punto)
            self.recargar_tabla()
            self.recargar_grafica()
        Popup_agregar_punto(self, callback)

    def borrar_punto(self, indice):
        self.puntos.pop(indice)
        self.recargar_tabla()
        self.recargar_grafica()

    def limpiar(self, *args):
        self.puntos = []
        self.recargar_tabla()
        self.recargar_grafica()

    def cargar_listas(self):
        self.listas['values'] = self.controlador.listas_puntos() 

    def guardar_puntos(self):
        if self.puntos:
            nombre = self.listas.get()
            self.controlador.guardar_puntos(nombre, self.puntos)
            self.cargar_listas()

    def cargar_puntos(self):
        puntos = self.controlador.cargar_puntos(self.listas.get())
        if puntos:
            self.puntos = puntos
            self.recargar_grafica()
            self.recargar_tabla()

    def recargar_tabla(self):
        # Sería ideal sólo insertar el punto en lugar de rehacer la
        # tabla pero no sé cómo tratar con el número i de cada punto
        self.tabla.limpiar_tabla()
        for i, point in enumerate(self.puntos):
            punto_trunco = tuple((round(x, ndigits=4) for x in point))
            self.tabla.agregar_entrada(str(i), *punto_trunco)

        self.tabla.desactivar_botones()

    def recargar_grafica(self):
        self.ax.clear()
        if self.puntos:
            puntosTranspuesto = list(zip(*self.puntos))
            self.ax.plot(*puntosTranspuesto[:3], color='lightcoral',
                        linewidth=1.5)
            self.ax.scatter(*puntosTranspuesto[:3], color='red')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.grafica.draw()


class Popup_agregar_punto(tk.Toplevel):

    def __init__(self, parent, callback_registro):
        super().__init__()
        self.parent = parent
        self.callback_registro = callback_registro
        self.wm_title("Agregar punto")

        self.definir_elementos()
        # Centrar pantalla
        x_pos = parent.winfo_rootx() + parent.winfo_width()//2 - 120
        y_pos = parent.winfo_rooty() + parent.winfo_height()//2 - 50
        self.geometry(f'+{x_pos}+{y_pos}')

    def definir_elementos(self):
        frame_xyz = ttk.Frame(self)
        frame_xyz.grid(column=0, row=0, sticky='ew')

        self.x_entry = Label_Entry(frame_xyz, label='x:', width=5,
                                   var_type='float', default_val=0.0)
        self.x_entry.grid(column=0, row=0)

        self.y_entry = Label_Entry(frame_xyz, label='y:', width=5,
                                   var_type='float', default_val=0.0)
        self.y_entry.grid(column=2, row=0)

        self.z_entry = Label_Entry(frame_xyz, label='z:', width=5,
                                   var_type='float', default_val=0.0)
        self.z_entry.grid(column=4, row=0)

        frame_tiempos = ttk.Frame(self)
        frame_tiempos.grid(column=0, row=1)

        self.t_t_entry = Label_Entry(frame_tiempos, width=5,
                                     label="Tiempo transición (s):",
                                     var_type='float', default_val=1.0,
                                     restr_positiv=True, non_zero=True)
        self.t_t_entry.grid(column=0, row=0)

        self.t_s_entry = Label_Entry(frame_tiempos, width=5,
                                     label="Tiempo estacionario (s):",
                                     var_type='float', default_val=1.0,
                                     restr_positiv=True, non_zero=True)
        self.t_s_entry.grid(column=0, row=1)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.agregar_punto)
        boton_aceptar.grid(column=0, row=2)

        for frame in [self, frame_xyz, frame_tiempos]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=3)

    def agregar_punto(self):
        x = self.x_entry.get()
        y = self.y_entry.get()
        z = self.z_entry.get()
        t_t = self.t_t_entry.get()
        t_s = self.t_s_entry.get()

        punto = [x, y, z, t_t, t_s]

        if not (None in punto):
            self.callback_registro(punto)
            self.destroy()


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaSelecPuntos(root)
    root.mainloop()