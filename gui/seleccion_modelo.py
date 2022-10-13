import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Pantalla, TablaYBotones, Label_Entry, TxtPopup
from gui.nuevo_modelo import Popup_agregar_modelo


class PantallaSelecModelo(Pantalla):

    def __init__(self, parent):
        super().__init__(parent, titulo="Seleccionar modelo")

    def definir_elementos(self):
        # Tabla principal
        columnas = (' nombre',
                    ' tipo',
                    ' entrenamientos')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(200, 120, 120),
                                   fn_doble_click=self.seleccionar_modelo)
        self.tabla.grid(column=0, row=0, sticky='nsew',
                        padx=5, pady=5)

        for modelo in self.controlador.modelos:
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

        self.tabla.agregar_boton(text="Ver hparams",
                                 width=20,
                                 command=self.ver_hparams,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Parámetros de ajuste",
                                 width=20,
                                 command=self.ver_ajustes,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Abrir tensorboard",
                                 width=20,
                                 command=self.abrir_tensorboard,
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
                                   modelo.cls_id,
                                   len(modelo.trains))

    def seleccionar_modelo(self, indice):
        self.controlador.seleccionar_modelo(indice)

    def agregar_modelo(self, *args):
        def callback(nombre, model_args):
            agregado = self.controlador.agregar_modelo(nombre, model_args)
            if agregado:
                self.agregar_modelo_tabla(self.controlador.modelos[-1])
                indice = len(self.controlador.modelos) - 1
                self.controlador.seleccionar_modelo(indice)
            return agregado
        Popup_agregar_modelo(self, callback)

    def copiar_modelo(self, indice):
        def callback(nombre):
            agregado = self.controlador.copiar_modelo(indice, nombre)
            if agregado:
                self.agregar_modelo_tabla(self.controlador.modelos[-1])
            return agregado
        Popup_copiar_modelo(self, callback)

    def ver_hparams(self, indice):
        hparams = self.controlador.modelos[indice].kwargs
        text = str().join(f"{key} : {val}\n" for key,val in hparams.items())
        TxtPopup(self, title="Parámetros", text=text,
                 width= 24, height= 12)

    def ver_ajustes(self, indice):
        trains = self.controlador.modelos[indice].trains
        text = str()
        for i, train in enumerate(trains):
            text += "="*24 + f"\n Ajuste {i+1} \n" + "="*24

            for etapa, args in train.items():
                text += "\n" + etapa + "\n"
                for key,val in args.items():
                    text += f"  {key} : {val}\n"

            text += "\n"

        if len(trains) == 0:
            text = "Sin entrenamientos registrados"

        TxtPopup(self, title="Entrenamientos", text=text,
                 width= 32, height= 16)

    def abrir_tensorboard(self, indice):
        self.controlador.seleccionar_modelo(indice)
        self.controlador.abrir_tensorboard(ver_todos=True)

    def eliminar_modelo(self, indice):
        if tk.messagebox.askyesno("Eliminar?",
                                  "Eliminar modelo?"):
            self.tabla.tabla.delete(self.tabla.tabla.focus())
            self.controlador.cerrar_tensorboard()
            self.controlador.eliminar_modelo(indice)
            self.tabla.desactivar_botones()

    def actualizar(self):
        super().actualizar()
        self.tabla.limpiar_tabla()
        self.tabla.desactivar_botones()
        for modelo in self.controlador.modelos:
            self.agregar_modelo_tabla(modelo)

    def en_cierre(self):
        self.controlador.cerrar_tensorboard()
        self.controlador.guardar()


class Popup_copiar_modelo(tk.Toplevel):

    def __init__(self, parent, callback):
        super().__init__()
        self.parent = parent
        self.callback = callback

        self.wm_title("Copiar modelo")
        self.definir_elementos()
        # Centrar pantalla
        x_pos = parent.winfo_rootx() + parent.winfo_width()//2 - 120
        y_pos = parent.winfo_rooty() + parent.winfo_height()//2 - 50
        self.geometry(f'+{x_pos}+{y_pos}')

    def definir_elementos(self):
        self.nom_entry = Label_Entry(self, label="Nombre", 
                                var_type='str', width=20)
        self.nom_entry.grid(column=0, row=0)

        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.destroy)
        boton_cancelar.grid(column=0, row=2)

        boton_aceptar = ttk.Button(self, text="Agregar",
                                   command=self.copiar_modelo)
        boton_aceptar.grid(column=1, row=2, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def copiar_modelo(self):
        nombre = self.nom_entry.get()
        if nombre != '':
            agregado = self.callback(nombre)
            if agregado:
                self.destroy()


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaSelecModelo(root)
    root.mainloop()