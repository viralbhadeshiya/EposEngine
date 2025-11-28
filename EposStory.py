import json
import os
import ollama
from PIL import Image
from typing import Dict, Optional
import re

class EposStory:
    def __init__(self, panels_path: str, output_path: str, episode: int, model_name: str = "qwen3-vl:latest"):
        """
        Initialize EposStory for generating video and audio prompts from comic panels.
        
        Args:
            panels_path: Path to directory containing panel images
            output_path: Path where transcript JSON will be saved
            episode: Episode number for naming output files
            model_name: Ollama model name (e.g., "qwen2.5-vl", "qwen2-vl", "qwen3-vl:2b-instruct")
                       Make sure the model is pulled in ollama: ollama pull <model_name>
        """
        print("Starting storyline process ...")
        self.panels_path = panels_path
        self.output_path = output_path
        self.episode = episode
        self.model_name = model_name
        self.transcript = {}

    def __validate_image(self, image_path: str) -> bool:
        """
        Validate that image file exists and is readable
        """
        try:
            if not os.path.exists(image_path):
                return False
            # Try to open and verify it's a valid image
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            print(f"Error validating image {image_path}: {e}")
            return False

    def __create_analysis_prompt(self) -> str:
        """
        Create comprehensive prompt for video and audio generation
        """
        prompt = """Analyze this comic panel image in extreme detail. Your task is to create two detailed prompts that will be used by AI agents to generate video and audio.

        IMPORTANT: You must respond ONLY in valid JSON format with the following structure:
        {
            "image_prompt": "detailed video generation prompt here",
            "audio_prompt": "detailed audio generation prompt here"
        }

        For the IMAGE_PROMPT, include:
        - Complete visual description of the scene (characters, setting, environment, objects)
        - Character appearances, clothing, expressions, poses, and body language
        - Lighting conditions (bright, dim, dramatic, natural, etc.)
        - Color palette and mood
        - Camera angle and perspective (close-up, wide shot, bird's eye, etc.)
        - Motion and animation details (what should move, how it should move)
        - Background details and atmosphere
        - Any text or speech bubbles visible (extract the text content)
        - Panel composition and layout
        - Style and artistic direction (realistic, cartoon, cinematic, etc.)
        - Any special effects or visual elements needed
        - Duration and pacing suggestions for the video

        For the AUDIO_PROMPT, include:
        - All dialogue text extracted from speech bubbles (if any)
        - Voice characteristics:
        * Gender (male, female, neutral, or specific if identifiable)
        * Age range (child, teenager, young adult, middle-aged, elderly)
        * Voice quality (deep, high-pitched, raspy, smooth, gruff, gentle, authoritative, etc.)
        * Accent or regional dialect (if identifiable)
        * Emotional tone (angry, happy, sad, fearful, excited, calm, etc.)
        * Speaking style (whisper, shout, normal, fast, slow, etc.)
        - Sound effects needed (if any visible or implied):
        * Environmental sounds (wind, rain, footsteps, etc.)
        * Action sounds (punch, explosion, door slam, etc.)
        * Ambient sounds (crowd, machinery, nature, etc.)
        - Music style and mood (if applicable):
        * Genre (dramatic, action, suspense, emotional, etc.)
        * Intensity level
        * Tempo and rhythm
        - Audio quality specifications:
        * Clarity level
        * Volume dynamics
        * Any special audio effects needed

        Be extremely detailed and descriptive. Include every visual and audio element you can identify. The more detail you provide, the better the final video and audio will be.

        Now analyze the image and provide your response in the JSON format specified above:"""
                
        return prompt

    def __parse_response(self, response: str) -> Optional[Dict[str, str]]:
        """
        Parse the model response to extract image_prompt and audio_prompt
        Handles both JSON and markdown-wrapped JSON
        """
        try:
            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            # Try to find JSON object in the response
            json_match = re.search(r'\{.*?"image_prompt".*?"audio_prompt".*?\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate required fields
            if "image_prompt" in data and "audio_prompt" in data:
                return {
                    "image_prompt": data["image_prompt"].strip(),
                    "audio_prompt": data["audio_prompt"].strip()
                }
            else:
                print("Warning: Response missing required fields")
                return None
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response[:500]}...")
            return None
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None

    def __analyze_panel_with_ollama(self, image_path: str) -> Optional[Dict[str, str]]:
        """
        Send panel image to ollama qwen3-vl model for analysis
        """
        try:
            # Validate image
            if not self.__validate_image(image_path):
                print(f"Invalid image file: {image_path}")
                return None
            
            # Create prompt
            prompt = self.__create_analysis_prompt()
            
            # Call ollama with image
            print(f"Analyzing panel: {os.path.basename(image_path)}")
            
            # Ollama can accept image paths directly or base64
            # Using absolute path to ensure it works
            abs_image_path = os.path.abspath(image_path)
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [abs_image_path]
                    }
                ],
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent, detailed output
                    "num_predict": 2000  # Allow longer responses for detailed prompts
                }
            )
            
            # Extract response content
            response_text = response['message']['content']
            
            # Parse response
            parsed = self.__parse_response(response_text)
            
            if parsed:
                print(f"✓ Successfully analyzed {os.path.basename(image_path)}")
                return parsed
            else:
                print(f"✗ Failed to parse response for {os.path.basename(image_path)}")
                print(f"Raw response preview: {response_text[:200]}...")
                return None
                
        except Exception as e:
            print(f"Error analyzing panel {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def __save_transcript(self):
        """
        Save transcript to JSON file
        """
        try:
            output_file = os.path.join(self.output_path, f"episode_{self.episode}_transcript.json")
            os.makedirs(self.output_path, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.transcript, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Transcript saved to: {output_file}")
            print(f"Total panels processed: {len(self.transcript)}")
            
        except Exception as e:
            print(f"Error saving transcript: {e}")

    def transcribe_panels(self):
        """
        Loop over panels to build transcript with video and audio prompts
        """
        if not os.path.exists(self.panels_path):
            print(f"Error: Panels path does not exist: {self.panels_path}")
            return
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        panel_files = [
            f for f in os.listdir(self.panels_path)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        # Sort panels naturally (panel_0, panel_1, etc.)
        panel_files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        
        if not panel_files:
            print(f"No image files found in {self.panels_path}")
            return
        
        # For testing: only process the first panel
        panel_files = panel_files[:1]
        
        print(f"Found {len(panel_files)} panel(s) to process (TESTING MODE: processing first panel only)\n")
        
        # Process each panel
        for idx, panel_file in enumerate(panel_files, 1):
            panel_path = os.path.join(self.panels_path, panel_file)
            panel_name = os.path.splitext(panel_file)[0]
            
            print(f"[{idx}/{len(panel_files)}] Processing {panel_file}...")
            
            # Analyze panel with ollama
            analysis = self.__analyze_panel_with_ollama(panel_path)
            
            if analysis:
                self.transcript[panel_name] = {
                    "panel_file": panel_file,
                    "image_prompt": analysis["image_prompt"],
                    "audio_prompt": analysis["audio_prompt"],
                    "panel_number": idx - 1
                }
            else:
                # Store error entry
                self.transcript[panel_name] = {
                    "panel_file": panel_file,
                    "image_prompt": None,
                    "audio_prompt": None,
                    "panel_number": idx - 1,
                    "error": "Failed to analyze panel"
                }
        
        # Save transcript
        self.__save_transcript()
        
        return self.transcript