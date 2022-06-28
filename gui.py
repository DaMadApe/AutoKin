from tkinter import *
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np


p_data = np.load('sofa/p_out.npy')[:10]

"""
Replica de pant5.png
"""
root = Tk()
root.title('Selección de puntos')

mainframe = ttk.Frame(root, padding="16 16 16 16")
mainframe.grid(column=0, row=0, sticky=(N,W,E,S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

"""
Definición de elementos
"""
# Títulos
titulo = ttk.Label(mainframe, text='Selección de puntos')

titulo_grafica = ttk.Label(mainframe, text='Previsualiación')

# Gráfica
fig = Figure(figsize=(4,3), dpi=100)
ax = fig.add_subplot(projection='3d')
ax.scatter(p_data[:,0], p_data[:,1], p_data[:,2])
fig.tight_layout()
grafica = FigureCanvasTkAgg(fig, master=mainframe)
grafica.draw()

# Tabla de puntos
columnas=('x', 'y', 'z')
tabla = ttk.Treeview(mainframe, columns=columnas, show=('tree','headings'))
tabla.column('#0', width=30, anchor='w')
tabla.heading('#0', text='i')
for col in columnas:
    tabla.column(col, width=50)
    tabla.heading(col, text=col)

for i, point in enumerate(p_data):
    tabla.insert('', 'end', text=str(i), 
                 values=tuple((round(x, ndigits=4) for x in point)))

# Configuraciones
configs = ttk.Frame(mainframe)

config1_lbl = ttk.Label(configs, text='Config 1')
config1_lbl.grid(column=0, row=0, sticky=W)
config2_lbl = ttk.Label(configs, text='Config 2')
config2_lbl.grid(column=0, row=1, sticky=W)

config1_entry = ttk.Entry(configs, width=10)
config1_entry.grid(column=1, row=0, sticky=E)
config2_entry = ttk.Entry(configs, width=10)
config2_entry.grid(column=1, row=1, sticky=E)

# Botones
boton_regresar = ttk.Button(mainframe, text='Regresar')
boton_regresar.grid(column=0, row=4, sticky=W)

boton_ejecutar = ttk.Button(mainframe, text='Ejecutar')
boton_ejecutar.grid(column=1, row=4, sticky=E)

"""
Acomodos
"""
titulo.grid(column=0, row=0, sticky=W)
titulo_grafica.grid(column=1, row=1, sticky=W)
tabla.grid(column=0, row=2, sticky=N)
grafica.get_tk_widget().grid(column=1, row=2, rowspan=2)
configs.grid(column=0, row=3, sticky=(W,E,N,S))
# bot_regresar.grid(column=1, row=1)
# bot_ejecutar.grid(column=1, row=1)


for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

for child in configs.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()