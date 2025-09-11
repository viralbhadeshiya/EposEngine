"""
EposEngine core file
Pipeline Information:
Stage 1 - Retrive file from location mention (can be .cbr or .pdf format)
Stage 2 - Panel whole document, crop images from comic and order them as story line.
Stage 3 - 2 parellele process (WIP though might change this to serial afterwards)
            a. Turning images from STAGE 2 to alive image/animated video
            b. Text-to-Speech for STAGE 2 o/
Stage 4 - Rendering of Voice on top of the animated video
"""
import os
import re
from EposCore import EposCore
from EposStory import EposStory

comic_path = './input/Invincible_000.pdf'
output_path = './_Invincible_000/pages/'

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

if __name__ == "__main__":
    # Extract pages and panels from normal comic book pdf
    episode = 1
    # eposCoreObject = EposCore(input_path=comic_path, start=3, end=13, output_dir=output_path)
    # eposCoreObject.convert_pdf_to_images()
    # files = sorted(os.listdir(output_path), key=natural_sort_key)
    # for filename in files:
    #     if filename.endswith('.jpg'):
    #         full_path = os.path.join(output_path, filename)
    #         panels_path = eposCoreObject.extract_panel_from_pages(page_path=full_path, episode=episode)
    
    # Make storyline, basically extract dialog from panels, sort each dialog in each panel.
    panels_path = os.path.join(output_path, "panels-1")
    eposStory = EposStory(panels_path, output_path, episode)
    eposStory.transcribe_panels()
