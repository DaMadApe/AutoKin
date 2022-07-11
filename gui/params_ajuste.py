import tkinter as tk
from tkinter import ttk

from gui.gui_utils import Label_Entry


class PantallaConfigAjuste(ttk.Frame):

    def __init__(self, parent):
        super().__init__(parent, padding="16 16 16 16")
        self.grid(column=0, row=0, sticky='nsew')

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.title("Parámetros de aproximación")

        self.definir_elementos()


    def definir_elementos(self):
        # Selección tipo de modelo
        frame_tipo = ttk.Frame(self)
        frame_tipo.grid(column=0, row=0, sticky='nsew')

        label_tipo = ttk.Label(frame_tipo, text="Tipo de modelo")
        label_tipo.grid(column=0, row=0)

        self.combo_tipos = ttk.Combobox(frame_tipo, state='readonly')
        self.combo_tipos.grid(column=0, row=1)
        self.combo_tipos['values'] = ('MLP', 'ResNet')
        self.combo_tipos.bind('<<ComboboxSelected>>', self.definir_panel_hparams)

        # Parámetros de modelo
        self.frame_mod_params = ttk.LabelFrame(self, text='Parámetros de modelo')
        self.frame_mod_params.grid(column=1, row=0, sticky='nsew')
        place_label = ttk.Label(self.frame_mod_params, text="\n\n")
        place_label.grid(column=0, row=0)

        # Parámetros de ajuste
        frame_t_params = ttk.LabelFrame(self, text='Parámetros de ajuste')
        frame_t_params.grid(column=1, row=1, sticky='nsew')

        b_size_entry = Label_Entry(frame_t_params,
                                   label='Batch size',
                                   var_type='int', default_val=256,
                                   restr_positiv=True, non_zero=True)
        b_size_entry.grid(column=0, row=0)

        l_rate_entry = Label_Entry(frame_t_params,
                                   label='Learning rate',
                                   var_type='float', default_val=1e-3,
                                   restr_positiv=True, non_zero=True)
        l_rate_entry.grid(column=0, row=1)

        # Botones inferiores
        boton_cancelar = ttk.Button(self, text="Cancelar",
                                   command=self.cancelar)
        boton_cancelar.grid(column=0, row=2)

        boton_cancelar = ttk.Button(self, text="Ejecutar",
                                   command=self.cancelar)
        boton_cancelar.grid(column=1, row=2)

        for child in self.winfo_children():
            child.grid_configure(padx=10, pady=5)

        for child in frame_t_params.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Comportamiento al cambiar de tamaño
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

    def definir_panel_hparams(self, event):
        tipo_modelo = self.combo_tipos.get()

        for widget in self.frame_mod_params.winfo_children():
            widget.destroy()

        if tipo_modelo == 'MLP':
            n_capas_entry = Label_Entry(self.frame_mod_params,
                                        label='# de capas',
                                        var_type='int', default_val=3,
                                        restr_positiv=True, non_zero=True)
            n_capas_entry.grid(column=0, row=0)

            n_neur_entry = Label_Entry(self.frame_mod_params,
                                       label='# de neuronas/capa',
                                       var_type='int', default_val=10,
                                       restr_positiv=True, non_zero=True)
            n_neur_entry.grid(column=0, row=1)

            f_act_label = ttk.Label(self.frame_mod_params, text="Función de activación")
            f_act_label.grid(column=0, row=2)
            f_act_combo = ttk.Combobox(self.frame_mod_params,state='readonly')
            f_act_combo.grid(column=1, row=2)
            f_act_combo['values'] = ('relu', 'tanh')

        if tipo_modelo == 'ResNet':
            n_capas_entry = Label_Entry(self.frame_mod_params,
                                        label='# de bloques',
                                        var_type='int', default_val=3,
                                        restr_positiv=True, non_zero=True)
            n_capas_entry.grid(column=0, row=0)

            n_capas_entry = Label_Entry(self.frame_mod_params,
                                        label='# de capas/bloque',
                                        var_type='int', default_val=3,
                                        restr_positiv=True, non_zero=True)
            n_capas_entry.grid(column=0, row=1)

            n_neur_entry = Label_Entry(self.frame_mod_params,
                                       label='# de neuronas/capa',
                                       var_type='int', default_val=10,
                                       restr_positiv=True, non_zero=True)
            n_neur_entry.grid(column=0, row=2)

            f_act_label = ttk.Label(self.frame_mod_params, text="Función de activación")
            f_act_label.grid(column=0, row=3)
            f_act_combo = ttk.Combobox(self.frame_mod_params,state='readonly')
            f_act_combo.grid(column=1, row=3)
            f_act_combo['values'] = ('relu', 'tanh')

        for child in self.frame_mod_params.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def cancelar(self):
        pass


if __name__ == '__main__':

    root = tk.Tk()
    root.geometry('800x450+100+100')
    root.minsize(550,330)
    root.maxsize(1200,800)
    pant1 = PantallaConfigAjuste(root)
    root.mainloop()