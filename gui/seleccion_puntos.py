import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.gui_utils import Pantalla, Label_Entry, Popup, TablaYBotones
from gui.const import args_etapas


class PantallaSelecPuntos(Pantalla):

    def __init__(self, parent):
        self.puntos = []

        self.dataset_check_var = tk.IntVar(value=0)
        self.ajuste_check_var = tk.IntVar(value=0)

        super().__init__(parent, titulo="Selección de puntos")

    def definir_elementos(self):
        columnas = ('i', 'x', 'y', 'z', 'ts')
        self.tabla = TablaYBotones(self, botones_abajo=True,
                                   columnas=columnas,
                                   anchos=(30, 60, 60, 60, 60),
                                   fn_doble_click=self.modificar_punto)
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
                                          rowspan=4, padx=5, pady=5)
        self.recargar_grafica()

        # Check para ver puntos de datasets pasados como referencia
        check_but = ttk.Checkbutton(self,
                                    text="Mostrar puntos de muestras anteriores",
                                    variable=self.dataset_check_var,
                                    command=self.recargar_grafica)
        check_but.grid(column=0, row=2, sticky='w')

        # Check para seleccionar ajuste continuo
        check_but = ttk.Checkbutton(self,
                                    text="Ajustar a nuevas muestras",
                                    variable=self.ajuste_check_var)
        check_but.grid(column=0, row=3, sticky='w')

        # Botones
        boton_regresar = ttk.Button(self, text="Regresar",
                                    command=self.parent.regresar)
        boton_regresar.grid(column=0, row=4, sticky='w')

        boton_ejecutar = ttk.Button(self, text="Ejecutar",
                                    command=self.ejecutar)
        boton_ejecutar.grid(column=1, row=4, sticky='e')

        # Agregar pad a todos los widgets
        for frame in [self]: #, frame_configs]:
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
        Popup_asignar_punto(self, callback)

    def modificar_punto(self, indice):
        def callback(punto):
            self.puntos.pop(indice)
            self.puntos.insert(indice, punto)
            self.recargar_tabla()
            self.recargar_grafica()
        Popup_asignar_punto(self, callback,
                            punto_prev=self.puntos[indice])

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
        nombre = self.listas.get()
        if self.puntos and nombre:
            self.controlador.guardar_puntos(nombre, self.puntos)
            self.cargar_listas()

    def cargar_puntos(self):
        puntos = self.controlador.cargar_puntos(self.listas.get())
        if puntos:
            self.puntos = puntos
            self.recargar_grafica()
            self.recargar_tabla()

    def ejecutar(self):
        if self.puntos:
            self.controlador.set_trayec(self.puntos,
                                        bool(self.ajuste_check_var.get()))
            self.parent.avanzar()

    def recargar_tabla(self):
        # Sería ideal sólo insertar el punto en lugar de rehacer la
        # tabla pero no sé cómo tratar con el número i de cada punto
        self.tabla.limpiar_tabla()
        for i, point in enumerate(self.puntos):
            punto_trunco = tuple((round(x, ndigits=4) for x in point))
            self.tabla.agregar_entrada(str(i+1), *punto_trunco)

        self.tabla.desactivar_botones()

    def recargar_grafica(self):
        self.ax.clear()
        if self.puntos:
            puntosTranspuesto = list(zip(*self.puntos))
            self.ax.plot(*puntosTranspuesto[:3], color='lightcoral',
                        linewidth=1.5)
            self.ax.scatter(*puntosTranspuesto[:3], color='red')

        if bool(self.dataset_check_var.get()):
            datasets = self.controlador.get_datasets()
            if datasets:
                max_points = 1000/len(datasets)
                for d_set in datasets.values():
                    d_set.apply_p_norm = False
                    # Limitar puntos mostrados
                    step = max(1, int(len(d_set)/max_points))
                    # Convertir tensor a np.array y acomodarlo
                    p_set = d_set[::step][1].numpy().transpose()
                self.ax.scatter(*p_set, color='royalblue')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.grafica.draw()


class Popup_asignar_punto(Popup):

    def __init__(self, parent, callback, punto_prev=None):
        self.callback = callback
        if punto_prev is None:
            punto_prev = [0., 0., 0., 1.]
        self.punto_prev = punto_prev
        super().__init__(title="Asignar punto", parent=parent)

    def definir_elementos(self):
        x_val, y_val, z_val, t_s_val = self.punto_prev

        frame_xyz = ttk.Frame(self)
        frame_xyz.grid(column=0, row=0, sticky='ew')

        self.x_entry = Label_Entry(frame_xyz, label='x:', width=5,
                                   var_type='float', default_val=x_val)
        self.x_entry.grid(column=0, row=0)

        self.y_entry = Label_Entry(frame_xyz, label='y:', width=5,
                                   var_type='float', default_val=y_val)
        self.y_entry.grid(column=2, row=0)

        self.z_entry = Label_Entry(frame_xyz, label='z:', width=5,
                                   var_type='float', default_val=z_val)
        self.z_entry.grid(column=4, row=0)

        frame_tiempos = ttk.Frame(self)
        frame_tiempos.grid(column=0, row=1)

        self.t_s_entry = Label_Entry(frame_tiempos, width=5,
                                     label="Tiempo estacionario (s):",
                                     var_type='float', default_val=t_s_val,
                                     restr_positiv=True, non_zero=True)
        self.t_s_entry.grid(column=0, row=0)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.agregar_punto)
        boton_aceptar.grid(column=0, row=2)

        for frame in [self, frame_xyz, frame_tiempos]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=3)

        self.bind('<Return>', self.agregar_punto)

    def agregar_punto(self, *args):
        x = self.x_entry.get()
        y = self.y_entry.get()
        z = self.z_entry.get()
        t_s = self.t_s_entry.get()

        punto = [x, y, z, t_s]

        if not (None in punto):
            self.callback(punto)
            self.destroy()


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaSelecPuntos(root)
    root.mainloop()