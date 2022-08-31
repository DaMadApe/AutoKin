import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Label_Entry, TablaYBotones
from gui.robot_database import UIController
from gui.nuevo_robot import Popup_agregar_robot


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
        # Tabla principal
        # columnas = reg.__repr__
        columnas = (' nombre',
                    ' tipo',
                    ' # de modelos',
                    ' # de actuadores')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(160, 120, 120, 120),
                                   fn_doble_click=self.seleccionar_robot)
        self.tabla.grid(column=0, row=0, sticky='nsew',
                        padx=5, pady=5)

        for robot in self.controlador.robots:
            self.agregar_robot_tabla(robot)

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
                                 command=self.ver_modelos,
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
                                         command=self.parent.regresar)
        self.boton_regresar.grid(column=0, row=1, sticky='e',
                                 padx=(0,10))

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def agregar_robot_tabla(self, robot):
        self.tabla.agregar_entrada(robot.nombre,
                                   robot.robot.__class__.__name__,
                                   len(robot.modelos),
                                   f"q = {robot.robot.n}")

    def agregar_robot(self, *args):
        def callback(nombre, robot):
            agregado = self.controlador.agregar_robot(nombre, robot)
            if agregado:
                self.controlador.guardar()
                self.agregar_robot_tabla(self.controlador.robots[-1])
            return agregado
        Popup_agregar_robot(self, callback)

    def seleccionar_robot(self, indice):
        self.controlador.robots.seleccionar(indice)
        self.controlador.guardar()

    def copiar_robot(self, indice):
        def callback(nombre):
            agregado = self.controlador.robots.copiar(indice, nombre)
            if agregado:
                self.controlador.guardar()
                self.agregar_robot_tabla(self.controlador.robots[-1])
            return agregado
        Popup_copiar_robot(self, callback)

    def ver_modelos(self, indice):
        self.controlador.robots.seleccionar(indice)
        self.controlador.guardar()
        self.parent.avanzar()

    def configurar_robot(self, indice):
        # Abrir interfaz de calibración
        pass

    def eliminar_robot(self, indice):
        self.controlador.robots.eliminar(indice)
        self.controlador.guardar()
        self.tabla.tabla.delete(self.tabla.tabla.focus())


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
                                   command=self.destroy)
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
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaSelecRobot(root)
    root.mainloop()