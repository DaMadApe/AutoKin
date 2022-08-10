import tkinter as tk
from tkinter import ttk

from gui.gui_utils import TablaYBotones
from gui.robot_database import UIController
from gui.nuevo_modelo import Popup_agregar_modelo


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
        # Tabla principal
        columnas = (' nombre',
                    ' tipo',
                    ' épocas')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(200, 120, 120),
                                   fn_doble_click=self.seleccionar_modelo)
        self.tabla.grid(column=0, row=0, sticky='nsew',
                        padx=5, pady=5)

        for modelo in self.controlador.modelos():
            self.agregar_modelo_tabla(modelo)

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

        # Botón de regresar pantalla
        self.boton_regresar = ttk.Button(self, text="Regresar",
                                         width=20,
                                         command=self.parent.regresar)
        self.boton_regresar.grid(column=0, row=1, sticky='e',
                                 padx=(0,10))

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def agregar_modelo_tabla(self, modelo):
        self.tabla.agregar_entrada(modelo.nombre,
                                   modelo.modelo.__class__.__name__,
                                   modelo.quick_log['epocs'])

    def seleccionar_modelo(self, indice):
        self.controlador.modelos().seleccionar(indice)
        self.controlador.guardar()

    def agregar_modelo(self, *args):
        def callback(nombre, modelo):
            agregado = self.controlador.agregar_modelo(nombre, modelo)
            if agregado:
                self.controlador.guardar()
                self.agregar_modelo_tabla(self.controlador.modelos()[-1])
            return agregado
        Popup_agregar_modelo(self, callback)

    def copiar_modelo(self, indice):
        def callback(nombre):
            agregado = self.controlador.modelos().copiar(indice, nombre)
            if agregado:
                self.controlador.guardar()
                self.agregar_modelo_tabla(self.controlador.modelos()[-1])
            return agregado
        # Popup_copiar_modelo(self, callback)

    def ver_log(self, indice):
        pass

    def eliminar_modelo(self, indice):
        self.controlador.modelos().eliminar(indice)
        self.controlador.guardar()
        self.tabla.tabla.delete(self.tabla.tabla.focus())