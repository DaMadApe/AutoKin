import builtins

import tkinter as tk
from tkinter import ttk


class Label_Entry:
    """
    Composición de ttk.Label y ttk.Entry con validación

    Requiere un acomodo tipo grid en el widget sobre el
    que se dibuja, ocupa 2 posiciones por default
    (vertical => rowspan=2; ~vertical => colspan=2).
    Ocupa 3 posiciones si se declara post_label
    """

    def __init__(self, parent, label:str,
                 var_type:str, default_val=None,
                 restr_positiv=False, non_zero=False,
                 vertical=False, post_label:str=None,
                 **entry_kwargs):

        self.var_type = var_type
        self.var_fun = getattr(builtins, var_type)
        self.pos_var = restr_positiv
        self.non_zero = non_zero
        self.vertical = vertical

        self.label = ttk.Label(parent, text=label)
        self.entry = ttk.Entry(parent, **entry_kwargs)

        self.post_label = None
        if post_label is not None:
            self.post_label = ttk.Label(parent, text=post_label)

        if default_val is not None:
            assert type(default_val)==self.var_fun
            self.entry.insert(0, default_val)

    def _validate_float(self, x):
        try:
            float(x)
            return True
        except:
            return False

    def valid(self, x:str):
        valid = True
        if self.var_type == 'float':
            valid &= self._validate_float(x)
        elif self.var_type == 'int':
            valid &= x.isnumeric()
        if valid:
            if self.pos_var:
                valid &= self.var_fun(x) >= 0
            if self.non_zero:
                valid &= self.var_fun(x) != 0
        return valid

    def get(self):
        entry_val = self.entry.get()
        if self.valid(entry_val):
            self.entry['style'] = 'TEntry'
            return self.var_fun(entry_val)
        else:
            self.entry['style'] = 'Red.TEntry'
            return None

    def grid(self, column, row, label_sticky='w',
             entry_sticky='we', **both_kwargs):
        c,r = (0,1) if self.vertical else (1,0)
        self.label.grid(column=column+0, row=row+0,
                        sticky=label_sticky, **both_kwargs)
        self.entry.grid(column=column+c, row=row+r,
                        sticky=entry_sticky, **both_kwargs)
        if self.post_label is not None:
            self.post_label.grid(column=column+2*c,
                                 row=row+2*r, sticky='w')


class TablaYBotones(ttk.Frame):
    """
    Widget compuesto de tabla + botones

    Parent (Widget): Objeto sobre el que se dibuja el Frame
    Columnas (Tuple[Str]) : Encabezados de la tabla
    anchos (Tuple[Int]) : Lista de ancho en pix para cada columna
    botones_abajo (Bool) : Activar para botones horizontales bajo
        la tabla, desactivar para botones verticales al lado
    fn_doble_click (Callable) : Función llamada cuando se hace
        doble click a un elemento de la tabla
    """

    def __init__(self, parent, columnas, anchos,
                 botones_abajo=False, fn_doble_click=None):
        super().__init__(parent)
        self.columnas = columnas
        self.anchos = anchos
        self.botones_abajo = botones_abajo
        self.fn_doble_click = fn_doble_click
        self.botones_vinculados = []
        self._definir_elementos()

    def _definir_elementos(self):

        self.tabla = ttk.Treeview(self, columns=self.columnas[1:], 
                                  show=('tree','headings'))
        self.tabla.grid(column=0, row=0, sticky='nsew')

        self.tabla.column('#0', width=self.anchos[0], anchor='w')
        self.tabla.heading('#0', text=self.columnas[0], anchor='w')
        for i, col in enumerate(self.columnas[1:]):
            self.tabla.column(col, width=self.anchos[i+1])
            self.tabla.heading(col, text=col, anchor='w')

        self.tabla.bind('<ButtonRelease-1>', self._clickTabla)
        self.tabla.bind('<Escape>', self._escaparTabla)
        if self.fn_doble_click is not None:
            self.tabla.bind('<Double-1>', self._dobleClickTabla)

        # Scrollbar de tabla
        vscroll = ttk.Scrollbar(self, command=self.tabla.yview)
        self.tabla.config(yscrollcommand=vscroll.set)
        vscroll.grid(column=1, row=0, sticky='ns')

        self.frame_botones = ttk.Frame(self)
        if self.botones_abajo:
            self.frame_botones.grid(column=0, row=1,
                                    columnspan=2, sticky='nsew')
        else:
            self.frame_botones.grid(column=2, row=0, sticky='nsew')

        self.rowconfigure(0, weight=2)
        self.columnconfigure(0, weight=2)

    def _clickTabla(self, event):
        if self.tabla.focus() != '':
            self.activar_botones()

    def _dobleClickTabla(self, event):
        self.fn_doble_click(self.indice_actual())

    def _escaparTabla(self, event):
        for elem in self.tabla.selection():
            self.tabla.selection_remove(elem)
        self.desactivar_botones()

    def _config_botones(self, activar:bool):
        estado = 'normal' if activar else 'disabled'
        for boton in self.botones_vinculados:
            boton['state'] = estado

    def _incluir_indice(self, fun):
        def aux():
            idx = self.indice_actual()
            return fun(idx)
        return aux

    def activar_botones(self):
        self._config_botones(activar=True)

    def desactivar_botones(self):
        self._config_botones(activar=False)

    def indice_actual(self):
        seleccion = self.tabla.focus()
        if seleccion == '':
            cur_idx = len(self.tabla.get_children())
        else:
            cur_idx = self.tabla.index(seleccion)
        return cur_idx

    def limpiar_tabla(self):
        self.tabla.delete(*self.tabla.get_children())

    def agregar_boton(self, rojo=False, activo_en_seleccion=False,
                      incluir_indice=True, padx=5, pady=5,
                      **btn_kwargs):
        if incluir_indice:
            btn_kwargs['command'] = self._incluir_indice(btn_kwargs['command'])

        boton = ttk.Button(self.frame_botones, **btn_kwargs)
        if rojo:
            boton['style'] = 'Red.TButton'
        if activo_en_seleccion:
            boton['state'] = 'disabled'
            self.botones_vinculados.append(boton)

        n_botones = len(self.frame_botones.winfo_children())

        c, r = (n_botones, 0) if self.botones_abajo else (0, n_botones)
        boton.grid(column=c, row=r, padx=padx, pady=pady)

    def agregar_entrada(self, *entrada):
        self.tabla.insert('', 'end', text=entrada[0],
                          values=entrada[1:])