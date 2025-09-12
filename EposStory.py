import json
import os
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en')
class EposStory:
    def __init__(self, panels_path: str, output_path: str, episode: int):
        print("Starting storyline process ...")
        self.panels_path = panels_path
        self.output_path = output_path
        self.episode = episode

    def __extract_text_from_panels(self, image_name: str):
        file_path = os.path.join(self.panels_path, image_name)
        results = ocr.ocr(file_path)
        lines = []
        for line in results[0]:
            box, (text, confidence) = line
            if confidence > 0.6:
                lines.append(text)

        return lines


    def transcribe_panels(self):
        "Loop over panels to build transcript"
        transcript = {}
        for panels in os.listdir(self.panels_path):
            panel_name = os.path.splitext(panels)[0]
            lines = self.__extract_text_from_panels(panels)
            if lines:
                transcript[panel_name] = [
                    {"line": line, "speaker": None} for line in lines
                ]
        with open(os.path.join(self.output_path, f"transcript_episode_{self.episode}.json"), "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)