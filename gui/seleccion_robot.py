import tkinter as tk
from tkinter import ttk

from gui.nuevo_robot import Popup_agregar_robot
from gui.gui_utils import Label_Entry, TablaYBotones
from gui.robot_database import UIController

save_dir = 'gui/app_data/robotRegs'


class PantallaSelecRobot(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Seleccionar robot")

        self.controlador = UIController()

        self.definir_elementos()

    def definir_elementos(self):
        style= ttk.Style()
        style.configure('Red.TButton', background='#FAA')
        style.map('Red.TButton', background=[('active', '#F66')])

        # Tabla principal
        # columnas = reg.__repr__
        columnas = (' nombre', '# de modelos', '# de actuadores')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(200, 120, 120),
                                   fn_doble_click=self.configurar_robot)
        self.tabla.grid(column=0, row=0, sticky='nsew',
                        padx=5, pady=5)

        for robot in self.controlador.robots:
            self.tabla.agregar_entrada(robot.nombre,
                                       len(robot.model_ids), robot.robot.n)

        # Botones de tabla
        self.tabla.agregar_boton(text="Nuevo...",
                                 width=20,
                                 command=self.agregar_robot)

        self.tabla.agregar_boton(text="Seleccionar",
                                 width=20,
                                 command=self.seleccionar_robot,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Copiar",
                                 width=20,
                                 command=self.copiar_robot,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Ver modelos",
                                 width=20,
                                 command=self.parent.avanzar,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Configurar",
                                 width=20,
                                 command=self.configurar_robot,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Eliminar",
                                 width=20,
                                 command=self.eliminar_robot,
                                 activo_en_seleccion=True,
                                 rojo=True)

        # Botón de regresar pantalla
        self.boton_regresar = ttk.Button(self, text="Regresar",
                                         width=20,
                                         command=self.regresar)
        self.boton_regresar.grid(column=0, row=1, sticky='e',
                                 padx=(0,10))

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def agregar_robot(self, *args):
        def callback(nombre, robot):
            agregado = self.controlador.robots.agregar(nombre, robot)
            print(f"callback: {agregado}")
            if agregado:
                self.tabla.agregar_entrada(nombre, "No",
                                           f"n = {robot.n}")
            return agregado
        Popup_agregar_robot(self, callback)

    def seleccionar_robot(self, indice):
        self.controlador.robots.seleccionar(indice)

    def copiar_robot(self, indice):
        def callback(nombre):
            agregado = self.controlador.robots.copiar(indice, nombre)
            return agregado
        Popup_copiar_robot(self, callback)

    def ver_modelos(self, indice):
        pass

    def configurar_robot(self, indice):
        # Abrir interfaz de calibración
        pass

    def eliminar_robot(self, indice):
        #del self.robots[indice_actual]
        self.controlador.robots.eliminar(indice)
        self.tabla.tabla.delete(self.tabla.tabla.focus())

    def regresar(self):
        # self.controlador.save() # Mejor en cada método
        self.parent.regresar()


class Popup_copiar_robot(tk.Toplevel):

    def __init__(self, parent, callback):
        super().__init__()
        self.parent = parent
        self.callback = callback

        self.wm_title("Copiar robot")
        self.definir_elementos()
        # Centrar pantalla
        x_pos = parent.winfo_rootx() + parent.winfo_width()//2 - 120
        y_pos = parent.winfo_rooty() + parent.winfo_height()//2 - 50
        self.geometry(f'+{x_pos}+{y_pos}')

    def definir_elementos(self):
        self.nom_entry = Label_Entry(self, label="Nombre", 
                                var_type='str', width=20)
        self.nom_entry.grid(column=0, row=0)

        self.check_copia = ttk.Checkbutton(self, text="Copiar modelos")
        self.check_copia.grid(column=0, row=1, columnspan=2)

        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.parent.regresar)
        boton_cancelar.grid(column=0, row=2)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.copiar_robot)
        boton_aceptar.grid(column=1, row=2, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=3)

    def copiar_robot(self):
        nombre = self.nom_entry.get()
        if nombre != '':
            agregado = self.callback(nombre)
            if agregado:
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