#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import time
import pygame
import cairo
import pango
import pangocairo
import sys
import math
import PIL
from PIL import Image
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


width, height = 320,120
import os
import pandas as pd
import numpy as np

screen = pygame.display.set_mode((width, height))
empty = pygame.Surface((width, height))
y = list(',.0123456789-_|')
CHARMAP = [unichr(i) for i in range(2304,2432)] + [unichr(i) for i in range(65,90)] + y
'''
def draw(ctx):
    ctx.set_line_width(15)
    ctx.arc(320, 240, 200, 0, 2 * math.pi)

    #                   r    g  b    a
    ctx.set_source_rgba(0.6, 0, 0.4, 1)
    ctx.fill_preserve()

    #                   r  g     b    a
    ctx.set_source_rgba(0, 0.84, 0.2, 0.5)
    ctx.stroke()
'''    
    
def bgra_surf_to_rgba_string(cairo_surface):
    # We use PIL to do this
    img = Image.frombuffer(
        'RGBA', (cairo_surface.get_width(),
                 cairo_surface.get_height()),
        cairo_surface.get_data(), 'raw', 'BGRA', 0, 1)

    return img.tobytes('raw', 'RGBA', 0, 1)#img.tostring('raw', 'RGBA', 0, 1)



#context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

fontname = sys.argv[1] if len(sys.argv) >= 2 else "Sans"


pathf = "line_images/"
FONT_TYPE = ["Dekko", "Shobhika", "Yatra One", "Yantramanav", "Kalam", "Utsaah", "Tillana", "Teko", "Sura", "Siddhanta", "Sarpanch", "Sarala", "Sarai", "Sanskrit 2003", "Sanskrit Text", "Samyak Devanagari", "Samanata", "SakalBharati",  "Sahadeva", "Rozha One", "Rhodium Libre", "Rajdhani", "Poppins", "Nirmala UI", "Nakula", "Modak", "Lohit Devanagari", "Kokila", "Khand","Karma", "Hind", "Halant", "GIST-DVOTMohini", "GIST-DVOTKishor", "GISTOT-BRXVinit", "GISTOT-DGRDhruv", "Eczar", "Ek Mukta", "Gargi", "Chandas", "Biryani", "Asar", "Arya", "Amiko", "Amita", "Aparajita", "Akshar Unicode","Laila", "Kurale", "Noto Sans", "Mukta", "Gotu", "Pragati Narrow", "Baloo 2", "Baloo", "Martel Sans", "Khula", "Jaldi", "Glegoo", "Palanquin", "Palanquin Dark", "Cambay", "Kadwa", "Vesper Libre", "Sumana", "Ranga", "Sahitya"]
# FONT_TYPE = ["Dekko", "Shobhika"]
print(sorted(FONT_TYPE))
print(len(FONT_TYPE))
if not os.path.exists(os.getcwd()+"/label_data"):
    os.mkdir(os.getcwd()+"/"+"label_data")
with open("label_data/"+"annot_synthetic.txt", "w") as ft:
    for fontname in FONT_TYPE:
        with open('data_preparation/synthetic/sanskritdoc.txt') as f:
            lines = random.sample(f.readlines(),5000)
        url_name = 1
        for line in lines:
            if not os.path.exists(pathf+fontname.replace(" ", "")):
                os.makedirs(pathf+fontname.replace(" ", ""))
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1800, 80)#like pygame surface
            context = cairo.Context(surf)
            #draw(context)

            #draw a background rectangle:
            context.rectangle(0,0,1800,80)# like screen above
            context.set_source_rgb(1, 1, 1)
            context.fill()
            # Translates context so that desired text upperleft corner is at 0,0
            context.translate(20,20) # translate by 50 in x annd 25 in x axis

            pangocairo_context = pangocairo.CairoContext(context)# context has cariaContext surf, background rectangle, white filled, translated
            pangocairo_context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)# context transferred to pangocariocontext in last and this command set to antialias

            layout = pangocairo_context.create_layout()# create a layout wotj pccontext defined above   
            font = pango.FontDescription(fontname + " 25")#like pygame.freetype.Font   
            layout.set_font_description(font)# set font to layout defined above
            layout.set_text(u''+line)# set text as this to layout
            context.set_source_rgb(0, 0, 0)
            pangocairo_context.update_layout(layout)
            pangocairo_context.show_layout(layout)

            with open(pathf+fontname.replace(" ", "")+"/"+str(url_name)+ ".png", "wb") as image_file:
                # print pathf+"books_"+fontname.replace(" ", "")+"/"+"synthetic"+"/"+str(url_name)+ ".png"
                surf.write_to_png(image_file)
                line = line.upper()
                for i,c in enumerate(line):
                    if ord(c)>=65 and ord(c)<=90:
                        line = line.replace(c,"#")
                # print(line)
                line = line.strip()
                ft.write(os.getcwd()+"/line_images/"+fontname.replace(" ", "")+"/"+str(url_name)+ ".png"+" "+line)
                ft.write("\n")
                url_name = url_name + 1
                # print("Yes")
        print fontname
        



