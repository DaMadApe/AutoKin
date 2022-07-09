import os

import tkinter as tk
from tkinter import ttk
from tkinter import N, E, S, W

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

save_dir = 'app_data/trayec'

class PantallaSelecPuntos(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky=(N,W,E,S))

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title("Selección de puntos")

        self.puntos = []

        self.definir_elementos()

    def definir_elementos(self):

        # Tabla de puntos
        frame_tabla = ttk.Frame(self)
        frame_tabla.grid(column=0, row=0, sticky=(N,S,W,E))
        
        columnas=('x', 'y', 'z', 'tt', 'ts')
        self.tabla = ttk.Treeview(frame_tabla, columns=columnas, show=('tree','headings'))
        self.tabla.grid(column=0, row=0, sticky=(N,S,W,E))
        self.tabla.column('#0', width=30, anchor='w')
        self.tabla.heading('#0', text='i')
        for col in columnas:
            self.tabla.column(col, width=50)
            self.tabla.heading(col, text=col)

        self.tabla.bind('<ButtonRelease-1>', self.clickTabla)
        self.tabla.bind('<Double-1>', self.dobleClickTabla)
        self.tabla.bind('<Escape>', self.escaparTabla)

        # Scrollbar de tabla
        vscroll = ttk.Scrollbar(frame_tabla, command=self.tabla.yview)
        vscroll.grid(column=1, row=0, sticky=(N,S))
        self.tabla.config(yscrollcommand=vscroll.set)

        # Botones para agregar y borrar puntos
        frame_puntos = ttk.Frame(frame_tabla)
        frame_puntos.grid(column=0, row=1, sticky=S)

        boton_agregar = ttk.Button(frame_puntos, text="Agregar punto",
                                   command=self.dialogo_agregar_punto)
        boton_agregar.grid(column=0, row=0, padx=(0, 10))

        self.boton_borrar = ttk.Button(frame_puntos, text="Borrar punto",
                                       command=self.borrar_punto)
        self.boton_borrar.grid(column=1, row=0, padx=(0, 10))
        self.boton_borrar['state'] = 'disabled'

        boton_limpiar = tk.Button(frame_puntos, text="Limpiar",
                                   bg='#FAA', activebackground='#F66',
                                   command=self.limpiar)
        boton_limpiar.grid(column=2, row=0, padx=(30, 0))

        # Guardar/Cargar lista de puntos
        frame_guardar = ttk.Frame(frame_tabla)
        frame_guardar.grid(column=0, row=2, sticky=S)

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
        self.grafica.get_tk_widget().grid(column=1, row=0, sticky=N,
                                          rowspan=2, padx=5, pady=5)
        self.recargar_grafica()

        # Configuraciones del movimiento
        frame_configs = ttk.LabelFrame(self)
        frame_configs.grid(column=0, row=1, sticky=(W,E,N,S))

        config1_lbl = ttk.Label(frame_configs, text="Config 1")
        config1_lbl.grid(column=0, row=0, sticky=W)
        config2_lbl = ttk.Label(frame_configs, text="Config 2")
        config2_lbl.grid(column=0, row=1, sticky=W)

        config1_entry = ttk.Entry(frame_configs, width=10)
        config1_entry.grid(column=1, row=0, sticky=E)
        config2_entry = ttk.Entry(frame_configs, width=10)
        config2_entry.grid(column=1, row=1, sticky=E)

        # Botones
        boton_regresar = ttk.Button(self, text="Regresar")
        boton_regresar.grid(column=0, row=2, sticky=(W))

        boton_ejecutar = ttk.Button(self, text="Ejecutar")
        boton_ejecutar.grid(column=1, row=2, sticky=E)

        # Agregar pad a todos los widgets
        for frame in [self, frame_configs, frame_tabla]:
            for child in frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

        # Comportamiento al cambiar de tamaño
        frame_tabla.rowconfigure(0, weight=2)
        frame_tabla.rowconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)


    def dialogo_agregar_punto(self):
        Popup_agregar_punto(self)

    def limpiar(self):
        self.puntos = []
        self.recargar_tabla()
        self.recargar_grafica()

    def cargar_listas(self):
        self.listas['values'] = [os.path.splitext(n)[0] for n in os.listdir(save_dir)]

    def guardar_puntos(self):
        if self.puntos:
            nombre = self.listas.get()
            save_path =  os.path.join(save_dir, nombre)
            np.save(save_path, np.array(self.puntos))
            self.cargar_listas()

    def cargar_puntos(self):
        nombre = self.listas.get() + '.npy'
        load_path = os.path.join(save_dir, nombre)

        if nombre and os.path.exists(load_path):
            self.puntos = np.load(load_path).tolist()
            self.recargar_grafica()
            self.recargar_tabla()

    def agregar_punto(self, punto):
        seleccion = self.tabla.focus()
        if seleccion == '':
            indice_actual = len(self.puntos)
        else:
            indice_actual = self.tabla.index(seleccion)

        self.puntos.insert(indice_actual, punto)
        self.recargar_tabla()
        self.recargar_grafica()

    def borrar_punto(self):
        seleccion = self.tabla.focus()
        indice_actual = self.tabla.index(seleccion)

        self.puntos.pop(indice_actual)
        self.recargar_tabla()
        self.recargar_grafica()

    def recargar_tabla(self):
        # Sería ideal sólo insertar el punto en lugar de rehacer la
        # tabla pero no sé cómo tratar con el número i de cada punto
        self.tabla.delete(*self.tabla.get_children())
        for i, point in enumerate(self.puntos):
            punto_trunco = tuple((round(x, ndigits=4) for x in point))
            self.tabla.insert('', 'end', text=str(i), 
                              values=punto_trunco)

        self.boton_borrar['state'] = 'disabled'

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

    def clickTabla(self, event):
        if self.tabla.focus() != '':
            self.boton_borrar['state'] = 'normal'

    def dobleClickTabla(self, event):
        self.dialogo_agregar_punto()

    def escaparTabla(self, event):
        for elem in self.tabla.selection():
            self.tabla.selection_remove(elem)
        self.boton_borrar['state'] = 'disabled'


class Popup_agregar_punto(tk.Toplevel):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.wm_title("Agregar punto")
        self.definir_elementos()

    def definir_elementos(self):
        frame_xyz = ttk.Frame(self)
        frame_xyz.grid(column=0, row=0, sticky=(W,E))

        x_label = ttk.Label(frame_xyz, text="x:")
        x_label.grid(column=0, row=0)
        self.x_entry = ttk.Entry(frame_xyz, width=5)
        self.x_entry.grid(column=1, row=0)
        self.x_entry.insert(0, 0.)

        y_label = ttk.Label(frame_xyz, text="y:")
        y_label.grid(column=2, row=0)
        self.y_entry = ttk.Entry(frame_xyz, width=5)
        self.y_entry.grid(column=3, row=0)
        self.y_entry.insert(0, 0.)

        z_label = ttk.Label(frame_xyz, text="z:")
        z_label.grid(column=4, row=0)
        self.z_entry = ttk.Entry(frame_xyz, width=5)
        self.z_entry.grid(column=5, row=0)
        self.z_entry.insert(0, 0.)

        frame_tiempos = ttk.Frame(self)
        frame_tiempos.grid(column=0, row=1)

        t_t_label = ttk.Label(frame_tiempos, text="Tiempo transición (s):")
        t_t_label.grid(column=0, row=0, sticky=E)
        self.t_t_entry = ttk.Entry(frame_tiempos, width=5,
                                   validatecommand=self.validate_pos_float)
        self.t_t_entry.grid(column=1, row=0)
        self.t_t_entry.insert(0, 1.)

        t_s_label = ttk.Label(frame_tiempos, text="Tiempo estacionario (s):")
        t_s_label.grid(column=0, row=1, sticky=E)
        self.t_s_entry = ttk.Entry(frame_tiempos, width=5)
        self.t_s_entry.grid(column=1, row=1)
        self.t_s_entry.insert(0, 1.)

        boton_aceptar = ttk.Button(self, text="Agregar",
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
            self.error_msg.set("Números inválidos")

    def validate_float(self, x):
        try:
            float(x)
        except:
            return False
        else:
            return True

    def validate_pos_float(self, x):
        return self.validate_float(x) and float(x)>0


if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('800x450+100+100')
    root.minsize(550,330)
    root.maxsize(1200,800)
    pant5 = PantallaSelecPuntos(root)
    root.mainloop()