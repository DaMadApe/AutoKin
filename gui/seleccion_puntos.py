import tkinter as tk
from tkinter import ttk
from tkinter import N, E, S, W

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PantallaSelecPuntos(ttk.Frame):
    
    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.parent = parent
        self.parent.title('Selección de puntos')
        self.definir_elementos()

    def definir_elementos(self):
        self.grid(column=0, row=0, sticky=(N,W,E,S))
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)

        # Títulos
        titulo = ttk.Label(self, text='Selección de puntos')
        titulo.grid(column=0, row=0, sticky=W)

        titulo_grafica = ttk.Label(self, text='Previsualiación')
        titulo_grafica.grid(column=1, row=1, sticky=W)

        # Tabla de puntos
        columnas=('x', 'y', 'z')
        tabla = ttk.Treeview(self, columns=columnas, show=('tree','headings'))
        tabla.column('#0', width=30, anchor='w')
        tabla.heading('#0', text='i')
        for col in columnas:
            tabla.column(col, width=50)
            tabla.heading(col, text=col)

        for i, point in enumerate(p_data):
            tabla.insert('', 'end', text=str(i), 
                        values=tuple((round(x, ndigits=4) for x in point)))

        tabla.grid(column=0, row=2, sticky=N)

        # Gráfica
        fig = Figure(figsize=(4,3), dpi=100)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(p_data[:,0], p_data[:,1], p_data[:,2])
        fig.tight_layout()
        grafica = FigureCanvasTkAgg(fig, master=self)
        grafica.draw()
        grafica.get_tk_widget().grid(column=1, row=2, rowspan=2)

        # Configuraciones
        configs = ttk.Frame(self)
        configs.grid(column=0, row=3, sticky=(W,E,N,S))

        config1_lbl = ttk.Label(configs, text='Config 1')
        config1_lbl.grid(column=0, row=0, sticky=W)
        config2_lbl = ttk.Label(configs, text='Config 2')
        config2_lbl.grid(column=0, row=1, sticky=W)

        config1_entry = ttk.Entry(configs, width=10)
        config1_entry.grid(column=1, row=0, sticky=E)
        config2_entry = ttk.Entry(configs, width=10)
        config2_entry.grid(column=1, row=1, sticky=E)

        # Botones
        boton_regresar = ttk.Button(self, text='Regresar')
        boton_regresar.grid(column=0, row=4, sticky=W)

        boton_ejecutar = ttk.Button(self, text='Ejecutar')
        boton_ejecutar.grid(column=1, row=4, sticky=E)

        # Agregar pad a todos los elementos
        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

        for child in configs.winfo_children():
            child.grid_configure(padx=5, pady=5)



if __name__ == "__main__":

    import numpy as np

    p_data = np.load('sofa/p_out.npy')[:10]

    root = tk.Tk()

    pant5 = PantallaSelecPuntos(root)

    root.mainloop()