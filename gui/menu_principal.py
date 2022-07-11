from asyncore import close_all
import tkinter as tk
from tkinter import ttk


class PantallaMenuPrincipal(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title("AutoKin")

        self.parent = parent

        self.definir_elementos()

    def definir_elementos(self):

        style= ttk.Style()
        style.configure('Green.TLabel', foreground='#3A2')
        style.configure('Red.TLabel', foreground='#A11')

        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=1, row=1, sticky='nsew', padx=10, pady=10)

        # Botones izquierdos
        boton_seleccionar = ttk.Button(frame_botones, text="Seleccionar",
                                       width=20,
                                       command=self.seleccionar_robot)
        boton_seleccionar.grid(column=0, row=0)

        self.boton_entrenar = ttk.Button(frame_botones, text="Entrenar",
                                          width=20,
                                          command=self.entrenar)
        self.boton_entrenar.grid(column=0, row=1)
        self.boton_entrenar['state'] = 'disabled'

        self.boton_controlar = ttk.Button(frame_botones, text="Controlar",
                                          width=20,
                                          command=self.controlar)
        self.boton_controlar.grid(column=0, row=2)
        self.boton_controlar['state'] = 'disabled'

        for child in frame_botones.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Panel de información derecho
        frame_status = ttk.Frame(self)
        frame_status.grid(column=2, row=1, sticky='ns')

        titulo_robot = ttk.Label(frame_status, text="Robot",
                                 font=(13))
        titulo_robot.grid(column=0, row=0, columnspan=2)

        self.label_robot = ttk.Label(frame_status, text="Sin seleccionar")
        self.label_robot.grid(column=0, row=1)

        self.boton_config = ttk.Button(frame_status, text="config...",
                                       command=self.config_robot)
        self.boton_config.grid(column=1, row=1, sticky='e')
        self.boton_config['state'] = 'disabled'

        titulo_estado = ttk.Label(frame_status, text="Estado",
                                  font=(13))
        titulo_estado.grid(column=0, row=2, columnspan=2)

        self.label_sys1 = ttk.Label(frame_status, text="Medición posición")
        self.label_sys1.grid(column=0, row=3)

        self.label_status1 = ttk.Label(frame_status, text="  conectado ")
        self.label_status1.grid(column=1, row=3)
        self.label_status1['style'] = 'Green.TLabel'

        self.label_sys2 = ttk.Label(frame_status, text="Controlador robot")
        self.label_sys2.grid(column=0, row=4)

        self.label_status2 = ttk.Label(frame_status, text="desconectado")
        self.label_status2.grid(column=1, row=4)
        self.label_status2['style'] = 'Red.TLabel'

        for child in frame_status.winfo_children():
                child.grid_configure(padx=5, pady=5)

        for child in self.winfo_children():
                child.grid_configure(padx=5, pady=5)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(3, weight=2)
        frame_botones.columnconfigure(0, weight=1)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(2, weight=2)

    def seleccionar_robot(self):
        self.parent.seleccionar_robot()

    def entrenar(self):
        self.parent.entrenar_robot()

    def controlar(self):
        self.parent.controlar_robot()

    def config_robot(self):
        pass


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

    pant1 = PantallaMenuPrincipal(root)
    root.mainloop()