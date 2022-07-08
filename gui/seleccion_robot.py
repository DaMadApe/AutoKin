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


        self.definir_elementos()

    def definir_elementos(self):
        # Tabla principal
        columnas = ('modelo', 'actuadores')
        self.tabla = ttk.Treeview(self, columns=columnas, 
                                   show=('tree','headings'))
        self.tabla.grid(column=0, row=0, sticky=(W,E))

        self.tabla.column('#0', width=160, anchor='w')
        self.tabla.heading('#0', text='nombre')
        for col in columnas:
            self.tabla.column(col, width=120)
            self.tabla.heading(col, text=col)

        # self.tabla.bind('<ButtonRelease-1>', self.clickTabla)
        # self.tabla.bind('<Double-1>', self.dobleClickTabla)
        # self.tabla.bind('<Escape>', self.escaparTabla)

        # Botones de tabla
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=0, row=1)

        boton_n_robot = ttk.Button(frame_botones, text="Nuevo...",
                                   command=self.dialogo_agregar_robot)
        boton_n_robot.grid(column=0, row=1)

        boton_copiar = ttk.Button(frame_botones, text="Copiar",
                                  command=self.copiar_robot)
        boton_copiar.grid(column=1, row=1)
        boton_copiar['state'] = DISABLED

        boton_t_log = ttk.Button(frame_botones, text="Log ajuste",
                                 command=self.abrir_train_log)
        boton_t_log.grid(column=2, row=1)
        boton_t_log['state'] = DISABLED

        boton_config = ttk.Button(frame_botones, text="Configurar",
                                  command=self.configurar_robot)
        boton_config.grid(column=3, row=1)
        boton_config['state'] = DISABLED

        boton_eliminar = tk.Button(frame_botones, text="Eliminar",
                                   bg='#FAA', activebackground='#F66',
                                   command=self.eliminar_robot)
        boton_eliminar.grid(column=4, row=1)
        boton_eliminar['state'] = DISABLED

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)
        for child in frame_botones.winfo_children():
            child.grid_configure(padx=3, pady=3)

    def dialogo_agregar_robot(self):
        Popup_agregar_robot(self)

    def copiar_robot(self):
        pass

    def abrir_train_log(self):
        pass

    def configurar_robot(self):
        pass

    def eliminar_robot(self):
        pass


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
        pass

    def cancelar(self):
        pass


if __name__ == '__main__':

    root = tk.Tk()
    root.resizable(width=False, height=False)
    pant1 = PantallaSelecRobot(root)
    root.mainloop()