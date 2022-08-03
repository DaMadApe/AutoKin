import tkinter as tk
from tkinter import ttk

from gui.nuevo_robot import Popup_agregar_robot
from gui.gui_utils import Label_Entry, TablaYBotones
from gui.robot_database import RobotDatabase

save_dir = 'gui/app_data/robotRegs'


class PantallaSelecRobot(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        self.parent = parent
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.title("Seleccionar robot")

        self.robots = RobotDatabase(save_dir)

        self.definir_elementos()

    def definir_elementos(self):
        style= ttk.Style()
        style.configure('Red.TButton', background='#FAA')
        style.map('Red.TButton', background=[('active', '#F66')])

        # Tabla principal
        columnas = (' nombre', '# de modelos', '# de actuadores')
        self.tabla = TablaYBotones(self, columnas=columnas,
                                   anchos=(200, 120, 120),
                                   fn_doble_click=self.configurar_robot)
        self.tabla.grid(column=0, row=0, sticky='nsew',
                        padx=5, pady=5)

        for robot in self.robots:
            self.tabla.tabla.insert('','end', text=robot.nombre, 
                                    values=(len(robot.model_ids), robot.robot.n))

        # Botones de tabla
        self.tabla.agregar_boton(text="Nuevo...",
                                 width=20,
                                 command=self.dialogo_agregar_robot)

        self.tabla.agregar_boton(text="Seleccionar",
                                 width=20,
                                 command=self.seleccionar_robot,
                                 activo_en_seleccion=True)

        self.tabla.agregar_boton(text="Copiar",
                                 width=20,
                                 command=self.dialogo_copiar_robot,
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
                                         command=self.regresar)
        self.boton_regresar.grid(column=0, row=1, sticky='e',
                                 padx=(0,10))

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def agregar_robot(self, nombre, robot):
        agregado = self.parent.agregar_robot(nombre, robot)
        # if agregado:
        #     self.tabla.insert('', 'end', text=nombre,
        #                     values=('No', 'n = ?'))
        return agregado

    def copiar_robot(self, nombre):
        indice_actual = self.tabla.index(self.tabla.focus())
        agregado = self.parent.copiar_robot(indice_actual, nombre)
        return agregado

    def dialogo_agregar_robot(self, *args):
        Popup_agregar_robot(self)

    def dialogo_copiar_robot(self, *args):
        Popup_copiar_robot(self)

    def seleccionar_robot(self, idx):
        self.parent.seleccionar_robot(idx)

    def ver_modelos(self, idx):
        pass

    def configurar_robot(self, idx):
        # Abrir interfaz de calibración
        pass

    def eliminar_robot(self, idx):
        #del self.robots[indice_actual]
        self.parent.eliminar_robot(idx)
        self.tabla.tabla.delete(self.tabla.tabla.focus())

    def regresar(self):
        self.destroy()


class Popup_copiar_robot(tk.Toplevel):

    def __init__(self, parent):
        super().__init__()
        self.parent = parent

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
            agregado = self.parent.copiar_robot(nombre)
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