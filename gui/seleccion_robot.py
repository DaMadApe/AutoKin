import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Pantalla, Popup, Label_Entry, TablaYBotones
from gui.nuevo_robot import Popup_agregar_robot


class PantallaSelecRobot(Pantalla):

    def __init__(self, parent):
        super().__init__(parent, titulo="Seleccionar robot")

    def definir_elementos(self):
        # Tabla principal
        # columnas = reg.__repr__
        columnas = (' nombre',
                    ' tipo',
                    ' # de modelos',
                    ' # de actuadores')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(160, 120, 120, 120),
                                   fn_doble_click=self.controlador.seleccionar_robot)
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
                                 command=self.controlador.seleccionar_robot,
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
                                   robot.cls_id,
                                   len(robot.modelos),
                                   f"q = {robot.init_obj().n}")

    def agregar_robot(self, *args):
        def callback(nombre, robot_args):
            agregado = self.controlador.agregar_robot(nombre, robot_args)
            if agregado:
                self.agregar_robot_tabla(self.controlador.robots[-1])
            return agregado
        Popup_agregar_robot(self, callback)

    def copiar_robot(self, indice):
        def callback(nombre, copiar_modelos):
            agregado = self.controlador.copiar_robot(indice, nombre, copiar_modelos)
            if agregado:
                self.agregar_robot_tabla(self.controlador.robots[-1])
            return agregado
        Popup_copiar_robot(self, callback)

    def ver_modelos(self, indice):
        self.controlador.seleccionar_robot(indice)
        self.parent.avanzar()

    def configurar_robot(self, indice):
        # Abrir interfaz de calibración
        pass

    def eliminar_robot(self, indice):
        if tk.messagebox.askyesno("Eliminar?",
                                  "Eliminar robot y todos sus modelos?"):
            self.controlador.eliminar_robot(indice)
            self.tabla.tabla.delete(self.tabla.tabla.focus())
            self.tabla.desactivar_botones()

    def actualizar(self):
        super().actualizar()
        self.tabla.limpiar_tabla()
        self.tabla.desactivar_botones()
        for robot in self.controlador.robots:
            self.agregar_robot_tabla(robot)

    def en_cierre(self):
        self.controlador.cerrar_tensorboard()
        self.controlador.guardar()


class Popup_copiar_robot(Popup):

    def __init__(self, parent, callback):
        self.callback = callback
        super().__init__(title="Copiar robot", parent=parent)

    def definir_elementos(self):
        self.nom_entry = Label_Entry(self, label="Nombre", 
                                var_type='str', width=20)
        self.nom_entry.grid(column=0, row=0)

        self.check_var = tk.IntVar(value=0)
        check_copia = ttk.Checkbutton(self,
                                      text="Copiar modelos",
                                      variable=self.check_var)
        check_copia.grid(column=0, row=1, columnspan=2)

        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.destroy)
        boton_cancelar.grid(column=0, row=2)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.copiar_robot)
        boton_aceptar.grid(column=1, row=2, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=3)

        self.bind('<Return>', self.copiar_robot)

    def copiar_robot(self, *args):
        nombre = self.nom_entry.get()
        copiar_modelos = bool(self.check_var.get())
        if nombre != '':
            agregado = self.callback(nombre, copiar_modelos)
            if agregado:
                self.destroy()


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaSelecRobot(root)
    root.mainloop()