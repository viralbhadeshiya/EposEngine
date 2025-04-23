from pdf2image import convert_from_path
from typing import NewType
import cv2
import os

class EposCore:
    Panel_Boxes = NewType('Panel_Boxes', list[tuple[int, int, int, int]])

    def __init__(self, input_path: str, start: int, end: int, output_dir: str):
        """
            input_path - input path of comic in pdf format
            start - page number you want to start from
            end - page number till to wanna convert 
            output_dir - output directory path
        """
        self.input_path = input_path
        self.start = start
        self.end = end
        self.output_dir = output_dir
        self.panel_number = 0

    def __sort_boxes(self, boxes: Panel_Boxes) -> Panel_Boxes:
        row_threshold = 50  # Adjust based on comic layout
        boxes.sort(key=lambda b: (b[1] // row_threshold, b[0]))
        return boxes

    def convert_pdf_to_images(self) -> None:
        """
            Covert pdf files to image(it will only convert pages between start and end number of pages).
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            convert_from_path(pdf_path=self.input_path, first_page=self.start, last_page=self.end, output_folder=self.output_dir)
        except Exception as e:
            print('Issue while coverting pdf to images:', e)
        
    def extract_panel_from_pages(self, page_path: str, episode: int) -> None:
        """
            Extract panel from pages, It uses Contoure detection for auto panel detection
            page_path: page_path for input image
            episode: episode number for storing director purpose
        """
        try:
            img = cv2.imread(page_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w*h > 10000:
                    boxes.append((x, y, w, h))

            sorted_boxes = self.__sort_boxes(boxes)

            os.makedirs(f'{self.output_dir}/Episode_{episode}', exist_ok=True)
            for i, (x, y, w, h) in enumerate(sorted_boxes):
                panel = img[y:y+h, x:x+w]
                cv2.imwrite(f"{self.output_dir}/Episode_{episode}/{self.panel_number}-panel.png", panel)
                self.panel_number += 1
        except Exception as e:
            print("Issue detect while extracting panel from pages:", e)