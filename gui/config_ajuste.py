import tkinter as tk
from tkinter import ttk
from autokin.modelos import FKEnsemble

from gui.gui_utils import Pantalla, Label_Entry
from gui.const import args_etapas


class PantallaConfigAjuste(Pantalla):

    def __init__(self, parent):
        self.args_etapas = args_etapas
        self.arg_getters = {etapa: {} for etapa in self.args_etapas.keys()}
        self.check_buts = {}
        self.check_vars = {}
        super().__init__(parent, titulo="Configurar aproximación")

    def definir_elementos(self):
        # Checkboxes para etapas de entrenamiento
        frame_checks = ttk.Frame(self)
        frame_checks.grid(column=0, row=0, sticky='nsew')

        titulo_checks = ttk.Label(frame_checks,
                                  text="Seleccionar etapas de ajuste",
                                  font=(12))
        titulo_checks.grid(column=0, row=0, padx=5, pady=5)

        for i, etapa in enumerate(self.args_etapas.keys()):
            check_var = tk.IntVar()
            check_but = ttk.Checkbutton(frame_checks, text=etapa,
                                        variable=check_var,
                                        command=self.actualizar_tabs)
            check_but.grid(column=0, row=i+1,
                           padx=5, pady=5, sticky='w')
            self.check_buts[etapa] = check_but
            self.check_vars[etapa] = check_var

        # Configuración de cada etapa
        self.tabs_config = ttk.Notebook(self)
        self.tabs_config.grid(column=0, row=1, sticky='nsew')

        for etapa, args in self.args_etapas.items():
            frame = ttk.Frame(self.tabs_config)

            for i, (arg_name, entry_kwargs) in enumerate(args.items()):
                entry = Label_Entry(frame, **entry_kwargs)
                entry.grid(column=0, row=i, padx=5, pady=5)
                self.arg_getters[etapa][arg_name] = entry.get

            self.tabs_config.add(frame, text=etapa)

        # Ajuste inicial checado por default, no modificable
        self.check_vars['Ajuste inicial'].set(1)
        self.check_buts['Ajuste inicial']['state'] = 'disabled'
        # Muestreo activo condicionado a que el modelo sea FKEnsemble
        if not isinstance(self.controlador.modelo_s, FKEnsemble):
            self.check_buts['Ajuste dirigido']['state'] = 'disabled'

        self.actualizar_tabs()

        # Botones inferiores
        frame_botones = ttk.Frame(self)
        frame_botones.grid(column=0, row=2, sticky='we')
        frame_botones.columnconfigure(0, weight=1)

        boton_cancelar = ttk.Button(frame_botones,
                                    text="Cancelar",
                                    command=self.parent.regresar)
        boton_cancelar.grid(column=0, row=0, sticky='w')

        boton_ejecutar = ttk.Button(frame_botones,
                                    text="Aceptar",
                                    command=self.aceptar)
        boton_ejecutar.grid(column=1, row=0, sticky='e')

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    def actualizar_tabs(self):
        for i, var in enumerate(self.check_vars.values()):
            state = 'normal' if var.get()==1 else 'hidden'
            self.tabs_config.tab(i, state=state)

    def aceptar(self):
            train_kwargs = self.get_train_kwargs()
            if train_kwargs is not None:
                self.controlador.set_train_kwargs(train_kwargs)
                self.parent.avanzar()

    def get_train_kwargs(self):
        train_kwargs = {}
        for etapa, args_etapa in self.arg_getters.items():
            # Sólo añadir params de etapas seleccionadas
            if self.check_vars[etapa].get() == 1:
                train_kwargs[etapa] = {}
                for arg_name, get_fn in args_etapa.items():
                    arg_value = get_fn()
                    train_kwargs[etapa][arg_name] = arg_value

                    if arg_value is None:
                        return None

        return train_kwargs


if __name__ == '__main__':
    from gui_utils import MockInterfaz

    root = MockInterfaz()
    PantallaConfigAjuste(root)
    root.mainloop()