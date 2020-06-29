from pathlib import Path

import pandas as pd

from cjo.base.stringconstants import cat, supercat
from functions import dataframe_operations, tex_functions


class Hierarchy:

    def __init__(self, source):
        if isinstance(source, Path) or isinstance(source, str):
            source = dataframe_operations.import_sr(source)

        if isinstance(source, pd.Series):
            assert source.index.name == cat, f'Series\'s index should be named {cat}'
            assert source.name == supercat, f'Series should be named {supercat}'
            assert source.index.is_unique, 'Categories are in multiple super-categories'
            source = {sc: set(source[source == sc].index) for sc in source.unique()}
        elif isinstance(source, dict):
            assert len(set.union(*source.values())) == sum([len(v) for v in source.values()]), \
                'Categories are in multiple super-categories'
        elif isinstance(source, Hierarchy):
            source = source.hierarchy
        else:
            raise TypeError(f'Invalid source type :{type(source)}. Must be str, Path, Series, dict or Hierarchy')

        self.hierarchy = {k: sorted(v) for k, v in source.items()}

    @property
    def sc(self):
        return sorted(self.hierarchy.keys())

    @property
    def c(self):
        return sorted(sum(self.hierarchy.values(), []))

    def verify_categories(self, categories):
        return sorted(categories) == self.c

    def __getitem__(self, k):
        return self.hierarchy[k]

    @property
    def h(self):
        return len(self.hierarchy)

    @property
    def num_cat(self):
        return len(self.c)

    def __len__(self):
        return len(self.hierarchy)

    def to_tex_table(self, fn_out=None, standalone=False, **kwargs):

        df = pd.DataFrame(columns=['Categories'])
        import string
        for k, v in self.hierarchy.items():
            df.loc[k, 'Categories'] = ", ".join([string.capwords(vi).replace('Np', 'NP') for vi in v])
        df.index.name = 'Super-Category'
        df = df.reset_index(drop=False).sort_values('Super-Category')

        kwargs.setdefault('column_format', 'lp{3.5in}')
        kwargs.setdefault('caption', 'Hierarchy used in the experiments. (N)P stands for (Non-)Perishable')
        kwargs.setdefault('index', False)
        if standalone:
            string = tex_functions.df_2_standalone_latex_str(df, **kwargs)
        else:
            string = tex_functions.df_to_table(df, **kwargs)

        if fn_out is not None:
            with open(fn_out, 'w+') as wf:
                wf.write(string)
        else:
            return string

    def __eq__(self, other):
        if not set(self.sc) == set(other.sc):
            return False
        for sc in self.sc:
            if not set(self[sc]) == set(other[sc]):
                return False
        return True

    # def to_graphml(self, fn_out):
    #     node_width = 300
    #     node_height = 60
    #     sc_sc_x_sep = 30
    #     sc_h_y_sep = 60
    #     c_cs_x_sep = 30
    #     c_c_y_sep = 15
    #     edge_x_shift = -c_cs_x_sep // 2
    #
    #     g = Graph()
    #
    #     def add_node(name, x, y):
    #         g.add_node(name, x=str(x), y=str(y), width=str(node_width), height=str(node_height), shape_fill='#FFFFFF',
    #                    font_size='20')
    #
    #     x_hierarchy = (self.h * node_width + (self.h - 1) * sc_sc_x_sep) // 2 - node_width // 2
    #     add_node('Hierarchy', x_hierarchy, 0)
    #     for i, k in enumerate(self.sc):
    #         v = self[k]
    #         sc_x = (node_width + sc_sc_x_sep) * i
    #         sc_y = (node_height + sc_h_y_sep)
    #         add_node(k, sc_x, sc_y)
    #         g.add_edge('Hierarchy', k, path=[(x_hierarchy + node_width // 2, node_height + sc_h_y_sep // 2),
    #                                          (sc_x + node_width // 2, sc_y - sc_h_y_sep // 2)])
    #         for j, vj in enumerate(sorted(v)):
    #             c_x = sc_x + c_cs_x_sep
    #             c_y = sc_y + (node_height + c_c_y_sep) * (j + 1)
    #             add_node(vj, x=c_x, y=c_y)
    #             g.add_edge(k, vj, path=[(sc_x - edge_x_shift, sc_y + node_height // 2),
    #                                     (sc_x - edge_x_shift, c_y + node_height // 2)])
    #
    #     with open(fn_out, 'w+') as wf:
    #         wf.write(g.get_graph())
