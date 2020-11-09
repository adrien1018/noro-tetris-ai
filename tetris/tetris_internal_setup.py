#!/usr/bin/env python3

from distutils.core import setup, Extension

#args = ['-fsanitize=address', '-fsanitize=undefined']
#args = ['-fsanitize=address', '-fsanitize=pointer-compare']
args = []

name = 'Tetris_Internal'
module = Extension(name, sources = ['tetris.cpp'],
        extra_compile_args = args,
        extra_link_args = args)
setup(name = name, ext_modules = [module])
