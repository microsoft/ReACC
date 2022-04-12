# -*- coding:utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import re
import pickle
from tqdm import tqdm
import random
from collections import Counter, OrderedDict
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index,
                   traverse)
from textwrap import dedent


# ['nwo', 'path', 'language', 'identifier', 'parameters', 'argument_list', 'return_statement', 'docstring', 'docstring_summary', 'docstring_tokens', 'function', 'function_tokens']

def clean_docstring_comments(comment):
    comment = comment.strip().strip(""" "' """)
    comment = "\n".join(map(lambda s: s.lstrip("#"), comment.splitlines()))
    return dedent(comment)

class processor(object):
    def __init__(self, lang, code=None, remove_comments=False):
        LANGUAGE = Language('parser/my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        self.parser = [parser]
        self.lang = lang
        self.remove_comments = remove_comments
        self.preserve_words = set(["self", "super", "Exception", "__init__", "__main__"])
        if code is None:
            self.tree = None
        else:
            self.update(code)

    def update(self, code, function=False):
        if self.lang == "php":
            code = "<?php"+code+"?>"
        if self.lang == "java" and function:
            code = "class A{\n"+code+"\n}"
        com = True
        if self.remove_comments:
            com = False
            try:
                code = remove_comments_and_docstrings(code, self.lang)
            except Exception:
                com = True
        self.code = code
        self.code_bytes = code.encode("utf-8")
        self.tree = self.parser[0].parse(self.code_bytes)
        root_node = self.tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = self.code.split('\n')
        self.code_tokens = [index_to_code_token(x, code) for x in tokens_index]  
        self.index_to_code = OrderedDict()
        for idx, (index, code) in enumerate(zip(tokens_index, self.code_tokens)):
            self.index_to_code[index] = (idx, code)
        return com

    def get_doc(self):
        """
        For file level data, merge the doc in each function
        """
        self.functions = []
        self.get_func_nodes(self.tree.root_node)
        docs = ""
        for func_node in self.functions:
            body_node = func_node.children[-1]
            if body_node.children and body_node.children[0].children:
                if body_node.children[0].children[0].type in ["string", "comment"]:
                    docs += clean_docstring_comments(
                        self.span_select(body_node.children[0].children[0]),
                    ) + "<endofline>"
        return docs

    def get_func_nodes(self, root):
        """
        For both function level and file level data
        """
        for node in root.children:
            if node.type == "function_definition":
                self.functions.append(node)
            else:
                self.get_func_nodes(node)
                

    def load_names(self, path):
        vnames = pickle.load(open(os.path.join(path, "vnames.pkl"), "rb"))
        self.preserve_words = vnames["not_replace"]
        self.vnames = vnames["cands"]

    def span_select(self, *nodes, indent=False):
        if not nodes:
            return ""
        start, end = nodes[0].start_byte, nodes[-1].end_byte
        select = self.code_bytes[start:end].decode("utf-8")
        if indent:
            return " " * nodes[0].start_point[1] + select
        return select

    def get_func_name(self):
        """
        For function level data only
        """
        root_node = self.tree.root_node
        func_nodes = [node for node in root_node.children if node.type == "function_definition"]
        try:
            func_name = func_nodes[0].child_by_field_name("name")
        except IndexError:
            return ""
        return self.span_select(func_name)

    def get_var_names(self):
        root_node = self.tree.root_node
        vnames = set()
        self._get_var_names_from_node(root_node, vnames)
        return vnames

    def get_api_seq(self):
        root_node = self.tree.root_node
        api_seq = []
        self._get_api_seq(root_node, api_seq)
        return api_seq

    def _get_var_names_from_node(self, node, vnames, inactive=False):
        if len(node.children) > 0:
            if node.type in ["method_invocation"]:
                self._get_var_names_from_node(node.children[0], vnames, inactive)
                for child in node.children[1:]:
                    if child.type == "argument_list":
                        self._get_var_names_from_node(child, vnames, inactive)
                    else:
                        self._get_var_names_from_node(child, vnames, True)
            else:
                for child in node.children:
                    if node.type in ["field_access", "modifiers", "import_declaration", "package_declaration"]:
                        self._get_var_names_from_node(child, vnames, True)
                    else:
                        self._get_var_names_from_node(child, vnames, inactive)
        elif node.type == "identifier":
            if not inactive:
                vnames.add(self.span_select(node))

    def _get_api_seq(self, node, api_seq):
        if node.type == "method_invocation":
            obj = node.child_by_field_name("object")
            func = node.child_by_field_name("name")
            if obj:
                api_seq.append(self.span_select(obj) + "." + self.span_select(func))
            else:
                api_seq.append(self.span_select(func))
        else:
            for child in node.children:
                self._get_api_seq(child, api_seq)

    def process(self, ratio=0.5, indent=False, add_dead_code=True, cut_ratio=0.0, function=False):
        vnames = [x for x in self.get_var_names() if x not in self.preserve_words]
        vnames = random.sample(vnames, int(len(vnames)*ratio))
        cands = random.sample(self.vnames, len(vnames)+3)
        dead_vars = cands[-3:]
        if add_dead_code:
            deadcode = self.insert_dead_code(dead_vars)
        else:
            deadcode = None
        replaced = {v: c for v, c in zip(vnames, cands[:-3])}
        self.index_to_new_code = {}
        self._replace_var_names_from_node(self.tree.root_node, replaced)
        code_string = self.untokenize(indent, deadcode, True, cut_ratio=cut_ratio, function=function)
        return code_string

    def process_no_replace(self, indent=False, add_dead_code=True, cut_ratio=0.0, function=False):
        dead_vars = random.sample(self.vnames, 3)
        if add_dead_code:
            deadcode = self.insert_dead_code(dead_vars)
        else:
            deadcode = None
        code_string = self.untokenize(indent, deadcode, False, cut_ratio=cut_ratio, function=function)
        return code_string
    
    def insert_dead_code(self, v):
        # dead code types, vars that can't appear in original code
        # A = B, A
        # A.C(B, 1), A
        # A = B + C, AB
        # A = B(C), AB
        # A = B.C(), ABC
        # for (String i: A){\nB(C)\n}
        # A = [B for B in range(C)]
        # if (C){\nA = B()\n}
        dead_type = random.randrange(7)
        if dead_type == 0:
            return f"{v[0]} = {v[1]};\n"
        elif dead_type == 1:
            return f"{v[0]}.{v[2]}({v[1]}, 1);\n"
        elif dead_type == 2:
            return f"{v[0]} = {v[1]} + {v[2]};\n"
        elif dead_type == 3:
            return f"{v[0]} = {v[1]}({v[2]});\n"
        elif dead_type == 4:
            return f"{v[0]} = {v[1]}.{v[2]}();\n"
        elif dead_type == 5:
            return "for (String i: %s){\n%s(%s)\n;}\n"%(v[0], v[1], v[2])
        elif dead_type == 6:
            return "if (%s){\n%s = %s\n;}\n"%(v[2], v[0], v[1])

    def _replace_var_names_from_node(self, node, replaced):
        if len(node.children) > 0:
            for child in node.children:
                self._replace_var_names_from_node(child, replaced)
        elif node.type == "identifier":
            try:
                idf = self.index_to_code[(node.start_point, node.end_point)][1]
            except KeyError:
                idf = "None"
            if idf in replaced:
                self.index_to_new_code[(node.start_point, node.end_point)] = replaced[idf]

    def untokenize(self, indent=False, deadcode=None, replaced=False, cut_ratio=0.0, function=False, fix_cut_pos=False):
        code_string = ""
        prev_sp = None
        prev_ep = None
        prev_indent = 0
        indent_size = -1
        if function:
            total_line = list(self.index_to_code.keys())[-2][0][0]
        else:
            total_line = list(self.index_to_code.keys())[-1][0][0]
        insert_line = random.randint(total_line//5, total_line*4//5)
        cut = random.random() < cut_ratio
        if cut:
            if fix_cut_pos:
                cut_pos = (len(self.index_to_code)-4)//2
            else:
                cut_pos = random.randint((len(self.index_to_code)-4)//3, (len(self.index_to_code)-4)*2//3)
        if function:
            poses = list(self.index_to_code.keys())[3:-1]
        else:
            poses = list(self.index_to_code.keys())
        for ip, pos in enumerate(poses):
            sp = pos[0]
            ep = pos[1]
            if cut and ip >= cut_pos:
                break
            if replaced and pos in self.index_to_new_code:
                add_token = self.index_to_new_code[pos]
            else:
                add_token = self.index_to_code[pos][1]
            if prev_sp is None or (sp[0] == prev_ep[0] and sp[1] == prev_ep[1]):
                code_string += add_token
            elif sp[0] == prev_ep[0]:
                if code_string and code_string[-1] != " ":
                    code_string += " "
                code_string += add_token
            else:
                # if cut and cut_line >= 1 and cut_line <= prev_ep[0]:
                #     break
                if replaced and deadcode:
                    if insert_line <= prev_ep[0]:
                        code_string += "\n" + deadcode
                        insert_line = total_line+2
                if indent and add_token:
                    code_string += "\n"
                    omit = False
                    if sp[1] != prev_indent and prev_indent == 0 and indent_size == -1:
                        indent_size = sp[1] - prev_indent
                    if sp[1] - prev_indent > 0:
                        if sp[1] - prev_indent > 2 * indent_size:
                            omit = True
                        else:
                            for i in range(prev_indent, sp[1], indent_size):
                                code_string += "<INDENT>"
                    elif sp[1] - prev_indent < 0:
                        for i in range(sp[1], prev_indent, indent_size):
                            code_string += "<DEDENT>"
                    code_string += add_token
                    if not omit:
                        prev_indent = sp[1]
                else:
                    code_string += "\n"
                    code_string += " " if sp[1] else ""
                    code_string += add_token
            prev_sp, prev_ep = sp, ep
        return re.sub(re.compile("\s*\n"), "\n", code_string.lstrip()).replace("\n", "<endofline>")

    def convert_to_normal(self, code):
        lines = code.split("<endofline>")
        indent_size = 4
        indent = 0
        res = ""
        for line in lines:
            indent += line.count("<INDENT>")
            indent -= line.count("<DEDENT>")
            res += "\n" + " "*indent_size*indent + line.replace("<INDENT>", "").replace("<DEDENT>", "")
        return res




