import json
import os
import pytesseract

from PIL import Image

class EposStory:
    def __init__(self, panels_path: str, output_path: str, episode: int):
        print("Starting storyline process ...")
        self.panels_path = panels_path
        self.output_path = output_path
        self.episode = episode

    def __extract_text_from_panels(self, image_name: str):
        file_path = os.path.join(self.panels_path, image_name)
        img = Image.open(file_path).convert("RGB")
        config = "--psm 6"
        text = pytesseract.image_to_string(img, config=config)

        lines = [line.strip() for line in text.split("\n") if line.strip()]
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