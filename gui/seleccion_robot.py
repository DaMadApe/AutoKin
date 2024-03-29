import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Pantalla, Popup, Label_Entry, TablaYBotones
from gui.nuevo_robot import Popup_agregar_robot
from gui.popups_config import Popup_config_ext, Popup_config_rtb, Popup_config_sofa
from gui.robot_database import RoboReg


class PantallaSelecRobot(Pantalla):

    def __init__(self, parent):
        super().__init__(parent, titulo="Seleccionar robot")

    def definir_elementos(self):
        # Tabla principal
        # columnas = reg.__repr__
        columnas = (' nombre',
                    ' tipo',
                    ' # de modelos',
                    ' # de actuadores',
                    ' # de datasets')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(130, 100, 120, 120, 120),
                                   fn_doble_click=self.controlador.seleccionar_robot)
        self.tabla.grid(column=0, row=0, sticky='nsew',
                        padx=5, pady=5)

        for idx,_ in enumerate(self.controlador.robots):
            self.agregar_robot_tabla(idx)

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

    def agregar_robot_tabla(self, idx: int):
        robot = self.controlador.robots[idx]
        # HACK: Evitar inicialización de robot externo para obtener n de actuadores
        if robot.cls_id == 'Externo':
            n = robot.kwargs['n']
        else:
            n = robot.init_obj().n

        self.tabla.agregar_entrada(robot.nombre,
                                   robot.cls_id,
                                   len(robot.modelos),
                                   f"q = {n}",
                                   self.controlador.count_datasets(idx))

    def agregar_robot(self, *args):
        def callback(nombre, robot_args):
            agregado = self.controlador.agregar_robot(nombre, robot_args)
            if agregado:
                self.agregar_robot_tabla(-1)
            return agregado
        Popup_agregar_robot(self, callback)

    def copiar_robot(self, indice):
        def callback(nombre, copiar_modelos, copiar_datasets):
            agregado = self.controlador.copiar_robot(indice, nombre,
                                                     copiar_modelos,
                                                     copiar_datasets)
            if agregado:
                self.agregar_robot_tabla(-1)
            return agregado
        Popup_copiar_robot(self, callback)

    def ver_modelos(self, indice):
        self.controlador.seleccionar_robot(indice)
        self.parent.avanzar()

    def configurar_robot(self, indice):
        popups = {"Externo" : Popup_config_ext,
                  "Sim. RTB" : Popup_config_rtb,
                  "Sim. SOFA" : Popup_config_sofa}
        self.controlador.seleccionar_robot(indice)
        robot_cls = self.controlador.robot_reg_s.cls_id
        popups[robot_cls](self,
                          callback=self.controlador.config_robot,
                          robot=self.controlador.robot_s)

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
        for idx,_ in enumerate(self.controlador.robots):
            self.agregar_robot_tabla(idx)

    def en_cierre(self):
        self.controlador.cerrar_tensorboard()
        self.controlador.guardar_registros()


class Popup_copiar_robot(Popup):

    def __init__(self, parent, callback):
        self.callback = callback
        super().__init__(title="Copiar robot", parent=parent)

    def definir_elementos(self):
        self.nom_entry = Label_Entry(self, label="Nombre", 
                                var_type='str', width=20)
        self.nom_entry.grid(column=0, row=0)

        self.check_mod_var = tk.IntVar(value=0)
        check_modelos = ttk.Checkbutton(self,
                                        text="Copiar modelos",
                                        variable=self.check_mod_var)
        check_modelos.grid(column=0, row=1, columnspan=2, sticky='w')

        self.check_ds_var = tk.IntVar(value=0)
        check_datasets = ttk.Checkbutton(self,
                                         text="Copiar datasets",
                                         variable=self.check_ds_var)
        check_datasets.grid(column=0, row=2, columnspan=2, sticky='w')

        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.destroy)
        boton_cancelar.grid(column=0, row=3)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.copiar_robot)
        boton_aceptar.grid(column=1, row=3, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=3)

        self.bind('<Return>', self.copiar_robot)

    def copiar_robot(self, *args):
        nombre = self.nom_entry.get()
        copiar_modelos = bool(self.check_mod_var.get())
        copiar_datasets = bool(self.check_ds_var.get())
        if nombre != '':
            agregado = self.callback(nombre, copiar_modelos, copiar_datasets)
            if agregado:
                self.destroy()


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaSelecRobot(root)
    root.mainloop()