import os

import tkinter as tk
from tkinter import ttk
from tkinter import N, E, S, W, NORMAL, DISABLED


class PantallaSelecRobot(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky=(N,W,E,S))

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title("Seleccionar robot")

        self.robots = {}

        self.definir_elementos()

    def definir_elementos(self):
        # Tabla principal
        columnas = ('modelo', 'actuadores')
        self.tabla = ttk.Treeview(self, columns=columnas, 
                                   show=('tree','headings'))
        self.tabla.grid(column=0, row=0, sticky=(N,S,W,E))

        self.tabla.column('#0', width=240, anchor='w')
        self.tabla.heading('#0', text='nombre')
        for col in columnas:
            self.tabla.column(col, width=120)
            self.tabla.heading(col, text=col)

        self.tabla.bind('<ButtonRelease-1>', self.clickTabla)
        self.tabla.bind('<Double-1>', self.dobleClickTabla)
        self.tabla.bind('<Escape>', self.escaparTabla)

        # Botones de tabla
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=1, row=0, sticky=N)

        boton_n_robot = ttk.Button(frame_botones, text="Nuevo...",
                                   width=20,
                                   command=self.dialogo_agregar_robot)
        boton_n_robot.grid(column=0, row=0)

        self.boton_copiar = ttk.Button(frame_botones, text="Copiar",
                                       width=20,
                                       command=self.copiar_robot)
        self.boton_copiar.grid(column=0, row=1)
        self.boton_copiar['state'] = DISABLED

        self.boton_t_log = ttk.Button(frame_botones, text="Log ajuste",
                                      width=20,
                                      command=self.abrir_train_log)
        self.boton_t_log.grid(column=0, row=2)
        self.boton_t_log['state'] = DISABLED

        self.boton_config = ttk.Button(frame_botones, text="Configurar",
                                       width=20,
                                       command=self.configurar_robot)
        self.boton_config.grid(column=0, row=3)
        self.boton_config['state'] = DISABLED

        self.boton_eliminar = tk.Button(frame_botones, text="Eliminar",
                                        width=18,
                                        bg='#FAA', activebackground='#F66',
                                        command=self.eliminar_robot)
        self.boton_eliminar.grid(column=0, row=4)
        self.boton_eliminar['state'] = DISABLED

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)
        for child in frame_botones.winfo_children():
            child.grid_configure(padx=3, pady=3)

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def dialogo_agregar_robot(self):
        Popup_agregar_robot(self)

    def agregar_robot(self, nombre, n_q):
        self.robots[nombre] = {'n_q' : n_q}
        # Guardar/sobreescribir archivo de robot(s)

    def copiar_robot(self):
        indice_actual = self.tabla.index(self.tabla.focus())
        pass

    def abrir_train_log(self):
        pass

    def configurar_robot(self):
        # Abrir interfaz de calibración
        pass

    def eliminar_robot(self):
        pass

    def clickTabla(self, event):
        if self.tabla.focus() != '':
            self.boton_copiar['state'] = NORMAL
            self.boton_t_log['state'] = NORMAL
            self.boton_config['state'] = NORMAL
            self.boton_eliminar['state'] = NORMAL

    def dobleClickTabla(self, event):
        self.configurar_robot()

    def escaparTabla(self, event):
        for elem in self.tabla.selection():
            self.tabla.selection_remove(elem)
        self.boton_copiar['state'] = DISABLED
        self.boton_t_log['state'] = DISABLED
        self.boton_config['state'] = DISABLED
        self.boton_eliminar['state'] = DISABLED


class Popup_agregar_robot(tk.Toplevel):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.wm_title("Nuevo robot")
        self.definir_elementos()

    def definir_elementos(self):

        nom_label = ttk.Label(self, text="Nombre")
        nom_label.grid(column=0, row=0, sticky=W)

        self.nom_entry = ttk.Entry(self)
        self.nom_entry.grid(column=0, row=1, sticky=(E,W))

        n_q_label = ttk.Label(self, text="Número de actuadores")
        n_q_label.grid(column=0, row=2, sticky=W)

        self.n_q_entry = ttk.Entry(self)
        self.n_q_entry.grid(column=0, row=3, sticky=(E,W))

        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=0, row=4)

        boton_aceptar = ttk.Button(frame_botones, text="Agregar",
                                   command=self.agregar_robot)
        boton_aceptar.grid(column=0, row=0)

        boton_aceptar = ttk.Button(frame_botones, text="Cancelar",
                                   command=self.cancelar)
        boton_aceptar.grid(column=1, row=0)

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=3)

    def agregar_robot(self):
        nombre = self.nom_entry.get()
        n_q = self.n_q_entry.get()
        if n_q.isnumeric():
            n_q = int(n_q)
            self.parent.agregar_robot(nombre, n_q)
            self.destroy()

    def cancelar(self):
        self.destroy()


if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('800x450+100+100')
    root.minsize(550,330)
    root.maxsize(1200,800)
    pant1 = PantallaSelecRobot(root)
    root.mainloop()