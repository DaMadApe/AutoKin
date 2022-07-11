import os

import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Label_Entry

save_dir = 'app_data/robots'

class PantallaSelecRobot(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Seleccionar robot")

        self.robots = []

        self.definir_elementos()

    def definir_elementos(self):
        # Tabla principal
        columnas = ('modelo', 'actuadores')
        self.tabla = ttk.Treeview(self, columns=columnas, 
                                   show=('tree','headings'))
        self.tabla.grid(column=0, row=0, rowspan=2, sticky='nsew')

        self.tabla.column('#0', width=200, anchor='w')
        self.tabla.heading('#0', text=' nombre', anchor='w')
        for col in columnas:
            self.tabla.column(col, width=120)
            self.tabla.heading(col, text=col, anchor='w')

        self.tabla.bind('<ButtonRelease-1>', self.clickTabla)
        self.tabla.bind('<Double-1>', self.dobleClickTabla)
        self.tabla.bind('<Escape>', self.escaparTabla)

        # Botones de tabla
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=1, row=0, sticky='n')

        boton_n_robot = ttk.Button(frame_botones, text="Nuevo...",
                                   width=20,
                                   command=self.dialogo_agregar_robot)
        boton_n_robot.grid(column=0, row=0)

        self.boton_copiar = ttk.Button(frame_botones, text="Copiar",
                                       width=20,
                                       command=self.copiar_robot)
        self.boton_copiar.grid(column=0, row=1)

        self.boton_t_log = ttk.Button(frame_botones, text="Log ajuste",
                                      width=20,
                                      command=self.abrir_train_log)
        self.boton_t_log.grid(column=0, row=2)

        self.boton_config = ttk.Button(frame_botones, text="Configurar",
                                       width=20,
                                       command=self.configurar_robot)
        self.boton_config.grid(column=0, row=3)

        # TODO: Actualizar a ttk.Button, revisar SelecPuntos/Limpiar
        self.boton_eliminar = tk.Button(frame_botones, text="Eliminar",
                                        width=18,
                                        bg='#FAA', activebackground='#F66',
                                        command=self.eliminar_robot)
        self.boton_eliminar.grid(column=0, row=4)

        self.desactivar_botones()

        for child in frame_botones.winfo_children():
            child.grid_configure(padx=3, pady=3)


        self.boton_regresar = ttk.Button(self, text="Regresar",
                                         width=18,
                                         command=self.regresar)
        self.boton_regresar.grid(column=1, row=1)


        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)


        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def dialogo_agregar_robot(self):
        Popup_agregar_robot(self)

    def agregar_robot(self, nombre):
        for robot in self.robots:
            if robot['nombre'] == nombre:
                return False
        self.robots.append({'nombre': nombre})
        # pickle/json.save(self.robots)
        self.tabla.insert('', 'end', text=nombre,
                          values=('No', 'n = ?'))
        return True

    def copiar_robot(self):
        indice_actual = self.tabla.index(self.tabla.focus())
        pass

    def abrir_train_log(self):
        pass

    def configurar_robot(self):
        # Abrir interfaz de calibración
        pass

    def eliminar_robot(self):
        elem_actual = self.tabla.focus()
        indice_actual = self.tabla.index(elem_actual)
        
        self.robots.pop(indice_actual)
        self.tabla.delete(elem_actual)

    def regresar(self):
        self.destroy()

    def clickTabla(self, event):
        if self.tabla.focus() != '':
            self.activar_botones()

    def dobleClickTabla(self, event):
        self.configurar_robot()

    def escaparTabla(self, event):
        for elem in self.tabla.selection():
            self.tabla.selection_remove(elem)
        self.desactivar_botones()

    def config_botones(self, activar:bool):
        estado = 'normal' if activar else 'disabled'
        self.boton_copiar['state'] = estado
        self.boton_t_log['state'] = estado
        self.boton_config['state'] = estado
        self.boton_eliminar['state'] = estado

    def activar_botones(self):
        self.config_botones(activar=True)

    def desactivar_botones(self):
        self.config_botones(activar=False)


class Popup_agregar_robot(tk.Toplevel):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        self.wm_title("Nuevo robot")
        self.definir_elementos()
        # Centrar pantalla
        x_pos = parent.winfo_rootx() + parent.winfo_width()//2 - 120
        y_pos = parent.winfo_rooty() + parent.winfo_height()//2 - 50
        self.geometry(f'+{x_pos}+{y_pos}')

    def definir_elementos(self):

        self.nom_entry = Label_Entry(self, label="Nombre", 
                                var_type='str', width=20)
        self.nom_entry.grid(column=0, row=0)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.agregar_robot)
        boton_aceptar.grid(column=0, row=1)

        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.cancelar)
        boton_cancelar.grid(column=1, row=1, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=3)

    def agregar_robot(self):
        nombre = self.nom_entry.get()
        if nombre != '':
            agregado = self.parent.agregar_robot(nombre)
            if agregado:
                self.destroy()

    def cancelar(self):
        self.destroy()


if __name__ == '__main__':

    root = tk.Tk()
    root.minsize(550,330)
    root.maxsize(1200,800)

    win_width = 800
    win_height = 450
    x_pos = int(root.winfo_screenwidth()/2 - win_width/2)
    y_pos = int(root.winfo_screenheight()/2 - win_height/2)

    geom = f'{win_width}x{win_height}+{x_pos}+{y_pos}'
    root.geometry(geom)

    pant1 = PantallaSelecRobot(root)
    root.mainloop()