import builtins

import tkinter as tk
from tkinter import ttk
from tkinter import N, E, S, W


class Label_Entry: #(ttk.Frame):

    def __init__(self, parent, label:str,
                 var_type:str, default_val=None,
                 restr_positiv=False, non_zero=False,
                 vertical=False, post_label:str=None,
                 **entry_kwargs):
        #super().__init__(parent)

        style= ttk.Style()
        style.configure('Red.TEntry', foreground='red')

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

    def valid(self, x:str):
        valid = True
        if self.var_type == 'float':
            valid &= validate_float(x)
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

    def grid(self, column, row, label_sticky='w', entry_sticky='we'):
        c,r = (0,1) if self.vertical else (1,0)
        self.label.grid(column=column+0, row=row+0, sticky=label_sticky)
        self.entry.grid(column=column+c, row=row+r, sticky=entry_sticky)
        if self.post_label is not None:
            self.post_label.grid(column=column+2*c, row=row+2*r, sticky='w')



def validate_float(x):
    try:
        float(x)
        return True
    except:
        return False