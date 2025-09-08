from pdf2image import convert_from_path
from typing import NewType, Tuple
import cv2
import os
import numpy as np
from scipy import ndimage


class EposCore:
    Panel_Boxes = NewType('Panel_Boxes', list[tuple[int, int, int, int]])
    Rect = Tuple[int, int, int, int]

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
            images = convert_from_path(pdf_path=self.input_path, first_page=self.start, last_page=self.end)

            for i, image in enumerate(images):
                image.save(f"{self.output_dir}/page{i + 1}.jpg", "JPEG")
        except Exception as e:
            print('Issue while coverting pdf to images:', e)

    def __group_contiguous(self, idx: np.ndarray) -> List[Tuple[int, int]]:
        """
            Group contiguous indices into ranges.
        """
        if idx.size == 0:
            return []
        
        bands = []
        s = int(idx[0])
        p = s
        for v in map(int, idx[1:]):
            if v == p+1:
                p = v
            else:
                bands.append((s, p))
                s = v
                p = v
        bands.append((s, p))
        return bands

    def __columns_for_band(self, yo: int, y1: int) -> List[int]:
        band = white_mask[y0:y1, :]
        col_frac = band.mean(Axis=0)
        v_gutter_cols = np.where(col_frac > col_white_frac_thresh)[0]
        v_bands = self.__group_contiguous(v_gutter_cols)

    def __panel_rectangles_from_gutters(
        gray: np.ndarray,
        white_thresh: int = 245,
        row_white_frac_thresh: float = 0.92,
        col_white_frac_thresh: float = 0.93,
        min_rect_size_px: int = 40,
        region_mostly_white_cutoff: float = 0.90,
        inset_px: int = 4,
        row_grouping_fraction: float = 0.06,
    ) -> List[Rect]:
        """
            Extract panel rectangles from gutters in the image.
        """
        try:
            h, w = gray.shape[:2]

            # 1) White mask for gutter/background
            white_mask = (gray > white_thresh).astype(np.uint8)

            # 2) Close tiny gaps in gutters
            white_mask = cv2.morphologyEx(
                white_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), interactions=1
            )

            # 3) Horizontal gutters: with very high white fraction
            row_frac = white_mask.mean(axis=1)
            h_gutter_rows = np.where(row_frac > row_white_frac_thresh)[0]
            h_bands = self.__group_contiguous(h_gutter_rows)

            # Centerlines (gutters) + page borders
            h_lines = [0]
            for a ,b in h_bands:
                h_lines.append((a + b) // 2)
            h_lines.append(h)

            reacts: List[Rect] = []

        except Exception as e:
            print("Issue while extracting panel rectangles from gutters:", e)
        
    def extract_panel_from_pages(self, page_path: str, episode: int) -> None:
        """
            Extract panel from pages, It uses Contoure detection for auto panel detection
            page_path: page_path for input image
            episode: episode number for storing director purpose
        """
        try:
            img = cv2.imread(page_path)
            if img is None:
                raise ValueError(f"Failed to load image from {page_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            rects = 


        except Exception as e:
            print("Issue while extracting panel from pages:", e)