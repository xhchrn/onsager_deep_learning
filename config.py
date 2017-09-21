#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', 'True', 'TRUE', 1)

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg



