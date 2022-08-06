import tkinter as tk
from tkinter import ttk

from gui.gui_utils import TablaYBotones
from gui.robot_database import UIController


class PantallaSelecModelo(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Seleccionar modelo")

        self.controlador = UIController()

        self.definir_elementos()

    def definir_elementos(self):
        style= ttk.Style()
        style.configure('Red.TButton', background='#FAA')
        style.map('Red.TButton', background=[('active', '#F66')])

        # Tabla principal
        # TODO: Cambiar por modelos
        columnas = (' nombre', '# de modelos', '# de actuadores')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(200, 120, 120),
                                   fn_doble_click=self.configurar_robot)
        self.tabla.grid(column=0, row=0, sticky='nsew',
                        padx=5, pady=5)

        # TODO: Cambiar por modelos
        for robot in self.controlador.robots:
            self.tabla.agregar_entrada(robot.nombre,
                                       len(robot.model_ids), robot.robot.n)

        # Botones de self.tabla
        self.tabla.agregar_boton(text="Nuevo...",
                                 width=20,
                                 command=self.agregar_modelo)

        self.tabla.agregar_boton(text="Seleccionar",
                                 width=20,
                                 command=self.seleccionar_modelo,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Copiar",
                                 width=20,
                                 command=self.copiar_modelo,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Ver log",
                                 width=20,
                                 command=self.ver_log,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Eliminar",
                                 width=20,
                                 command=self.eliminar_modelo,
                                 activo_en_seleccion=True,
                                 rojo=True)