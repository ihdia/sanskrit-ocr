# -*- coding: utf-8 -*-
import time
import pygame
import cairo
import pango
import pangocairo
import sys
import math
import PIL
from PIL import Image

width, height = 320,120
import os
import pandas as pd
import numpy as np
DIRECTORY_PATH = os.getcwd()+"/books/books_3p"
screen = pygame.display.set_mode((width, height))
empty = pygame.Surface((width, height))
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


#get font families:

font_map = pangocairo.cairo_font_map_get_default()
families = font_map.list_families()

# to see family names:
print [f.get_name() for f in   font_map.list_families()]

#context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)



fontname = sys.argv[1] if len(sys.argv) >= 2 else "Sans"

# FONT_TYPE = [ "Inknut Antiqua", "Dekko", "Shobhika", "Yatra One", "Yantramanav", "Utsaah", "Tillana", "Teko", "Sura", "Siddhanta", "Sarpanch", "Sarala", "Sarai", "Sanskrit 2003", "Sanskrit Text", "Samyak Devanagari", "Samanata", "SakalBharati", "Sahitya", "Sahadeva", "Rozha One", "Rhodium Libre", "Rajdhani", "Poppins", "Nirmala UI", "Nakula", "Modak", "Lohit Devanagari", "Laila" "Kurale", "Kokila", "Khand","Karma", "Hind", "Halant", "GIST-DVOTMohini", "GIST-DVOTKishor", "GISTOT-BRXVinit", "GISTOT-DGRDhruv", "Eczar", "Ek Mukta", "Gargi", "Chandas", "Biryani", "Asar", "Arya", "Amiko", "Amita", "Aparajita", "Akshar Unicode"]
#FONT_TYPE = ["Laila", "Kurale", "Noto Sans", "Mukta", "Gotu", "Pragati Narrow", "Baloo 2", "Baloo", "Martel Sans", "Khula", "Jaldi", "Glegoo", "Palanquin", "Palanquin Dark", "Cambay", "Kadwa", "Vesper Libre", "Sumana", "Ranga", "Sahitya"]
FONT_TYPE = ["Yantramanav"]
for fontname in FONT_TYPE:
    print fontname
    
    
    for subdir, dirs, files in os.walk(DIRECTORY_PATH):
        index = subdir.rfind('/')
        index_book = subdir[0:index].rfind('/')
        for f in files:
            # print subdir
            if f == subdir[index+1:]+"_line.csv":
                df = pd.read_csv(subdir+"/"+f)
                for an, url_name in zip(df["Annotations"].astype(str), df["URL Name"].astype(str)):
                    if not os.path.exists(os.getcwd()+"/books/books_"+fontname):
                        os.mkdir(os.getcwd()+"/books/books_"+fontname)
                    if not os.path.exists(os.getcwd()+"/books/books_"+fontname+"/"+subdir[index_book+1:index]):
                        os.mkdir(os.getcwd()+"/books/books_"+fontname+"/"+subdir[index_book+1:index])
                    if not os.path.exists(os.getcwd()+"/books/books_"+fontname+"/"+subdir[index_book+1:index]+"/"+subdir[index+1:]):
                        os.mkdir(os.getcwd()+"/books/books_"+fontname+"/"+subdir[index_book+1:index]+"/"+subdir[index+1:])  
                        #print("Hi")
                    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1550, 80)#like pygame surface
                    context = cairo.Context(surf)
                    #draw(context)

                    #draw a background rectangle:
                    context.rectangle(0,0,1550,80)# like screen above
                    context.set_source_rgb(1, 1, 1)
                    context.fill()
                    # Translates context so that desired text upperleft corner is at 0,0
                    context.translate(20,20) # translate by 50 in x annd 25 in x axis

                    pangocairo_context = pangocairo.CairoContext(context)# context has cariaContext surf, background rectangle, white filled, translated
                    pangocairo_context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)# context transferred to pangocariocontext in last and this command set to antialias

                    layout = pangocairo_context.create_layout()# create a layout wotj pccontext defined above   
                    font = pango.FontDescription(fontname + " 25")#like pygame.freetype.Font   
                    layout.set_font_description(font)# set font to layout defined above
                    layout.set_text(u''+an)# set text as this to layout
                    context.set_source_rgb(0, 0, 0)
                    pangocairo_context.update_layout(layout)
                    pangocairo_context.show_layout(layout)

                    with open(os.getcwd()+"/books/books_"+fontname+"/"+subdir[index_book+1:index]+"/"+subdir[index+1:]+"/"+url_name+ ".png", "wb") as image_file:
                        print os.getcwd()+"/books/books_"+fontname+"/"+subdir[index_book+1:index]+"/"+subdir[index+1:]+"/"+url_name+ ".png"
                        surf.write_to_png(image_file)
                        # print("Yes")
