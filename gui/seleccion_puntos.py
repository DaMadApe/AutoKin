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

        self.puntos = None

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
                                   command=self.agregar_punto)
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


    def agregar_punto(self):
        popup = tk.Toplevel()
        popup.wm_title('Agregar punto')

        frame_xyz = ttk.Frame(popup)
        frame_xyz.grid(column=0, row=0, sticky=(W,E))
        x_label = ttk.Label(frame_xyz, text='x:')
        x_label.grid(column=0, row=0)
        # x_val = tk.DoubleVar()
        x_entry = ttk.Entry(frame_xyz, width=5) # , text=x_val)
        x_entry.grid(column=1, row=0)
        x_entry.insert(0, 0.)

        y_label = ttk.Label(frame_xyz, text='y:')
        y_label.grid(column=2, row=0)
        # y_val = tk.DoubleVar()
        y_entry = ttk.Entry(frame_xyz, width=5) # , text=y_val)
        y_entry.grid(column=3, row=0)
        y_entry.insert(0, 0.)

        z_label = ttk.Label(frame_xyz, text='z:')
        z_label.grid(column=4, row=0)
        # z_val = tk.DoubleVar()
        z_entry = ttk.Entry(frame_xyz, width=5) # , text=z_val)
        z_entry.grid(column=5, row=0)
        z_entry.insert(0, 0.)

        frame_tiempos = ttk.Frame(popup)
        frame_tiempos.grid(column=0, row=1)

        t_t_label = ttk.Label(frame_tiempos, text='Tiempo transición (s):')
        t_t_label.grid(column=0, row=0, sticky=E)
        t_t_val = tk.DoubleVar()
        t_t_val.set(1.)
        t_t_entry = ttk.Entry(frame_tiempos, width=5, textvariable=t_t_val)
        t_t_val = tk.DoubleVar()
        t_t_entry.grid(column=1, row=0)

        t_s_label = ttk.Label(frame_tiempos, text='Tiempo estacionario (s):')
        t_s_label.grid(column=0, row=1, sticky=E)
        t_s_val = tk.DoubleVar()
        t_s_val.set(1.)
        t_s_entry = ttk.Entry(frame_tiempos, width=5, textvariable=t_s_val)
        t_s_val = tk.DoubleVar()
        t_s_entry.grid(column=1, row=1)

        boton_aceptar = ttk.Button(popup, text='Agregar',
                                   command=self.guardar_lista_puntos)
        boton_aceptar.grid(column=0, row=2)

        for frame in [popup, frame_xyz, frame_tiempos]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

        


    def guardar_lista_puntos(self):
        pass

    def cargar_lista_puntos(self):
        for i, point in enumerate(p_data):
            self.tabla.insert('', 'end', text=str(i), 
                        values=tuple((round(x, ndigits=4) for x in point)))

        self.ax.scatter(*p_data.T, color='r')
        self.grafica.draw()


# class Popup_agregar_punto:



if __name__ == "__main__":

    import numpy as np

    p_data = np.load('sofa/p_out.npy')[:10]

    root = tk.Tk()
    pant5 = PantallaSelecPuntos(root)
    root.mainloop()