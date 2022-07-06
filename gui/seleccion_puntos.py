from cmath import tan
import tkinter as tk
from tkinter import ttk
from tkinter import N, E, S, W

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PantallaSelecPuntos(ttk.Frame):
    
    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky=(N,W,E,S))

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title('Selección de puntos')

        self.puntos = []

        self.definir_elementos()

    def definir_elementos(self):

        # Títulos
        titulo = ttk.Label(self, text='Selección de puntos')
        titulo.grid(column=0, row=0, sticky=W)

        titulo_grafica = ttk.Label(self, text='Previsualiación')
        titulo_grafica.grid(column=1, row=1, sticky=W)

        # Tabla de puntos
        frame_tabla = ttk.Frame(self)
        frame_tabla.grid(column=0, row=2, sticky=(W,E))
        
        columnas=('x', 'y', 'z', 'tt', 'ts')
        self.tabla = ttk.Treeview(frame_tabla, columns=columnas, show=('tree','headings'))
        self.tabla.grid(column=0, row=0, sticky=(W,E))
        self.tabla.column('#0', width=30, anchor='w')
        self.tabla.heading('#0', text='i')
        for col in columnas:
            self.tabla.column(col, width=50)
            self.tabla.heading(col, text=col)

        # Boton para agregar entrada
        boton_agregar = ttk.Button(frame_tabla, text='Agregar punto',
                                   command=self.nuevo_punto)
        boton_agregar.grid(column=0, row=1, sticky=(W,E))

        # Guardar/Cargar lista de puntos
        frame_guardar = ttk.Frame(frame_tabla)
        frame_guardar.grid(column=0, row=2)

        boton_guardar = ttk.Button(frame_guardar, text='Guardar',
                                   command=self.guardar_lista_puntos)
        boton_guardar.grid(column=0, row=0, padx=(0, 5))

        boton_cargar = ttk.Button(frame_guardar, text='Cargar',
                                  command=self.cargar_lista_puntos)
        boton_cargar.grid(column=1, row=0, padx=(0, 5))

        seleccion = tk.StringVar()
        lista = ttk.Combobox(frame_guardar, textvariable=seleccion)
        lista.grid(column=2, row=0, padx=(5, 0))

        # Gráfica
        fig = Figure(figsize=(4,3), dpi=100)
        self.ax = fig.add_subplot(projection='3d')
        fig.tight_layout()
        self.grafica = FigureCanvasTkAgg(fig, master=self)
        self.grafica.get_tk_widget().grid(column=1, row=2, sticky=N,
                                          padx=5, pady=5)

        # Configuraciones del movimiento
        frame_configs = ttk.Frame(self)
        frame_configs.grid(column=0, row=3, sticky=(W,E,N,S))

        config1_lbl = ttk.Label(frame_configs, text='Config 1')
        config1_lbl.grid(column=0, row=0, sticky=W)
        config2_lbl = ttk.Label(frame_configs, text='Config 2')
        config2_lbl.grid(column=0, row=1, sticky=W)

        config1_entry = ttk.Entry(frame_configs, width=10)
        config1_entry.grid(column=1, row=0, sticky=E)
        config2_entry = ttk.Entry(frame_configs, width=10)
        config2_entry.grid(column=1, row=1, sticky=E)

        # Botones
        boton_regresar = ttk.Button(self, text='Regresar')
        boton_regresar.grid(column=0, row=4, sticky=W)

        boton_ejecutar = ttk.Button(self, text='Ejecutar')
        boton_ejecutar.grid(column=1, row=4, sticky=E)

        # Agregar pad a todos los widgets
        for frame in [self, frame_configs, frame_tabla]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

    def nuevo_punto(self):
        popup = Popup_agregar_punto(self)

    def guardar_lista_puntos(self):
        pass

    def cargar_lista_puntos(self):
        for i, point in enumerate(p_data):
            self.tabla.insert('', 'end', text=str(i), 
                        values=tuple((round(x, ndigits=4) for x in point)))

        self.ax.scatter(*p_data.T, color='r')
        self.grafica.draw()

    def agregar_punto(self, punto):
        self.puntos.append(punto) # TODO: Meter en fila de tabla seleccionada

        self.tabla.delete(*self.tabla.get_children())
        for i, point in enumerate(self.puntos):
            self.tabla.insert('', 'end', text=str(i), 
                        values=tuple((round(x, ndigits=4) for x in point)))


class Popup_agregar_punto(tk.Toplevel):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.wm_title('Agregar punto')
        self.definir_elementos()

    def definir_elementos(self):
        frame_xyz = ttk.Frame(self)
        frame_xyz.grid(column=0, row=0, sticky=(W,E))

        x_label = ttk.Label(frame_xyz, text='x:')
        x_label.grid(column=0, row=0)
        self.x_entry = ttk.Entry(frame_xyz, width=5)
        self.x_entry.grid(column=1, row=0)
        self.x_entry.insert(0, 0.)

        y_label = ttk.Label(frame_xyz, text='y:')
        y_label.grid(column=2, row=0)
        self.y_entry = ttk.Entry(frame_xyz, width=5)
        self.y_entry.grid(column=3, row=0)
        self.y_entry.insert(0, 0.)

        z_label = ttk.Label(frame_xyz, text='z:')
        z_label.grid(column=4, row=0)
        self.z_entry = ttk.Entry(frame_xyz, width=5)
        self.z_entry.grid(column=5, row=0)
        self.z_entry.insert(0, 0.)

        frame_tiempos = ttk.Frame(self)
        frame_tiempos.grid(column=0, row=1)

        t_t_label = ttk.Label(frame_tiempos, text='Tiempo transición (s):')
        t_t_label.grid(column=0, row=0, sticky=E)
        self.t_t_entry = ttk.Entry(frame_tiempos, width=5,
                                   validatecommand=self.validate_pos_float)
        self.t_t_entry.grid(column=1, row=0)
        self.t_t_entry.insert(0, 1.)

        t_s_label = ttk.Label(frame_tiempos, text='Tiempo estacionario (s):')
        t_s_label.grid(column=0, row=1, sticky=E)
        self.t_s_entry = ttk.Entry(frame_tiempos, width=5)
        self.t_s_entry.grid(column=1, row=1)
        self.t_s_entry.insert(0, 1.)

        boton_aceptar = ttk.Button(self, text='Agregar',
                                   command=self.agregar_punto)
        boton_aceptar.grid(column=0, row=4)

        self.error_msg = tk.StringVar()
        self.error_msg.set(' ')
        error_label = ttk.Label(frame_tiempos, textvariable=self.error_msg,
                                foreground='#f00')
        error_label.grid(column=0, row=5)

        for frame in [self, frame_xyz, frame_tiempos]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

    def agregar_punto(self):
        x = self.x_entry.get()
        y = self.y_entry.get()
        z = self.z_entry.get()
        t_t = self.t_t_entry.get()
        t_s = self.t_s_entry.get()

        valid = True
        for var in [x,y,z]:
            valid &= self.validate_float(var)
        for var in [t_t, t_s]:
            valid &= self.validate_pos_float(var)

        if valid:
            punto = [x, y, z, t_t, t_s]
            punto = [float(val) for val in punto]
            self.parent.agregar_punto(punto)
            self.destroy()
        else:
            self.error_msg.set('Números inválidos')

    def validate_float(self, x):
        try:
            float(x)
        except:
            return False
        else:
            return True

    def validate_pos_float(self, x):
        return self.validate_float(x) and float(x)>0


if __name__ == "__main__":

    import numpy as np

    p_data = np.load('sofa/p_out.npy')[:10]

    root = tk.Tk()
    pant5 = PantallaSelecPuntos(root)
    root.mainloop()