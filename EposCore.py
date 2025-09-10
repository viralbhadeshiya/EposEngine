from PIL import Image
from pdf2image import convert_from_path
from typing import NewType, Tuple, List
import cv2
import os
import torch
import numpy as np
from realesrgan import RealESRGANer
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

    def __columns_for_band(self, yo: int, y1: int, w: int, white_mask: np.ndarray, col_white_frac_thresh: float) -> List[int]:
        band = white_mask[yo:y1, :]
        col_frac = band.mean(axis=0)
        v_gutter_cols = np.where(col_frac > col_white_frac_thresh)[0]
        v_bands = self.__group_contiguous(v_gutter_cols)
        v_lines = [0]

        for a, b in v_bands:
            v_lines.append((a + b) // 2)
        v_lines.append(w)
        return v_lines

    def __iou(self, a: Rect, b: Rect) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def __panel_rectangles_from_gutters(
        self,
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
                white_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
            )

            # 3) Horizontal gutters: with very high white fraction
            row_frac = white_mask.mean(axis=1)
            h_gutter_rows = np.where(row_frac > row_white_frac_thresh)[0]
            h_bands = self.__group_contiguous(h_gutter_rows)

            # Center lines (gutters) + page borders
            h_lines = [0]
            for a ,b in h_bands:
                h_lines.append((a + b) // 2)
            h_lines.append(h)

            rects: [self.Rect] = []

            # 4) Split each horizontal band by vertical gutters
            for yi in range(len(h_lines) - 1):
                y0, y1 = h_lines[yi], h_lines[yi + 1]
                if y1 - y0 < min_rect_size_px:
                    continue

                v_lines = self.__columns_for_band(y0, y1, w, white_mask, col_white_frac_thresh)

                for xi in range(len(v_lines) - 1):
                    x0, x1 = v_lines[xi], v_lines[xi + 1]
                    if x1 - x0 < min_rect_size_px:
                        continue
                    
                    # Reject mostly white regions (to handle blank panel boxes)
                    # Sometime opencv get some empty bos which are either in between panel or somewhere in picture it self
                    region = white_mask[y0:y1, x0:x1]
                    if region.mean() > region_mostly_white_cutoff:
                        continue

                    #Tighten a little inside to avoid gutters
                    rects.append(
                        (
                            max(x0 + inset_px, 0),
                            max(y0 + inset_px, 0),
                            min(x1 - inset_px, w),
                            min(y1 - inset_px, h),
                        )
                    )

            # 5) Deduplicate by IoU
            rects_sorted = sorted(rects, key=lambda r: (r[1], r[0]))
            filtered: List[self.Rect] = []
            for r in rects_sorted:
                if all(self.__iou(r, o) < 0.9 for o in filtered):
                    filtered.append(r)
            
            # 6) Reading order: top -> bottom, then left -> right within rows
            rows: List[List[self.Rect]] = []
            row_thresh_px = int(row_grouping_fraction * h)
            for r in filtered:
                placed = False
                for row in rows:
                    if abs(row[0][1] - r[1]) < row_thresh_px:
                        row.append(r)
                        placed = True
                        break
                if not placed: # (this logic will appends all panel which don;t make sense at the end. TODO: think some logic for that)
                    rows.append([r])
            
            ordered: List[self.Rect] = []
            for row in rows:
                ordered.extend(sorted(row, key=lambda r: r[0]))

            return ordered
        except Exception as e:
            print("Issue while extracting panel rectangles from gutters:", e)

    def __save_panels(
        self,
        image_path: str,
        rects: List[Rect],
        out_dir: str,
        prefix: str = "panel_",
        quality: int = 95
    ) -> List[str]:
        """ Crop and save each rectangles to disk; return saved file path"""
        os.makedirs(out_dir, exist_ok=True)
        pil = Image.open(image_path)
        paths = []
        for i, (x0, y0, x1, y1) in enumerate(rects, start=1):
            crop = pil.crop((x0, y0, x1, y1))
            p = os.path.join(out_dir, f"{prefix}{self.panel_number}.png")
            self.panel_number += 1
            crop.save(p, dpi=(300, 300))
            paths.append(p)
        return paths

    def __super_resolve(self, img: Image.Image, model, scale=4) -> Image.Image:
        """Upscale using Real-ESRGAN"""
        img = img.covert("RGB")
        sr_img = model.predict(np.array(img))
        return Image.fromarray(sr_img)

    def __pad_to_4k(self, img: Image.Image, bg_color="white") -> Image.Image:
        target_size = (3840, 2160)
        img.thumbnail(target_size, Image.LANCZOS)

        new_img = Image.new("RGB", target_size, bg_color)
        x = (target_size[0] - img.width) // 2
        y = (target_size[1] - img.height) // 2
        new_img.paste(img, (x, y))
        return new_img
        
    def __upscale_resolution(self, panel_paths: List[str]):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RealESRGAN(device, scale=4)
        model.load_weights("RealESRGAN_x4plus.pth")

        for i, path in enumerate(panel_paths, 1):
            img = Image.open(path)
            img_sr = self.__super_resolve(img, model, scale=4)

            img_4k = self.__pad_to_4k(img_sr, bg_color="white")
            img_4k.save(path)

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

            rects = self.__panel_rectangles_from_gutters(gray)

            panel_output_dir = os.path.join(self.output_dir, f"panels-{episode}")
            saved = self.__save_panels(page_path, rects, out_dir=panel_output_dir)

            self.__upscale_resolution(saved)

        except Exception as e:
            print("Issue while extracting panel from pages:", e)