# Human Emulation System (Coding Edition) [HES-CODE.py]

from gradio import Interface
from gradio.components import Textbox
import logging
import os
import tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer


# Updated HumanEmulationSystem class with separate functions for left and right hemispheres

class HumanEmulationSystem:
    def __init__(self):
        # Define configuration settings.
        self.DEBUG = True  # Set to True to show API calls

        # Configure logging.
        logging.basicConfig(level=logging.DEBUG if self.DEBUG else logging.INFO)
        self.chat_history = ""

        # Setup LLM configuration and key (replace with your access token)
        self.access_token = "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"

        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/stablecode-instruct-alpha-3b",
            use_auth_token=self.access_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablecode-instruct-alpha-3b",
            trust_remote_code=True,
            use_auth_token=self.access_token,
            torch_dtype="auto"
        )
        self.model.cuda()

        # Set cognitive contexts.
        self.context_left = "Analytic Logic, Data-Driven Thinking, Focusing on Facts and Evidence"
        self.context_right = "Creative Reasoning, Intuition, Symbolic Linking, Exploring Possibilities"
        self.context_mid = "Polymath, Seamless Viewpoint Integration, Focused on Essential Aspects"

        # Set moderator modifiers.
        self.format_mod = " (Keep your response short, on topic, well balanced and concise.) "
        self.negative_mod = "[DO NOT IDENTIFY AS an LLM, AI, language model, or AI language model]"
        self.modifiers = self.format_mod + self.negative_mod

    @staticmethod
    def chat_log(chat, prompt, mid_result):
        log = f"{chat}User(Input): {prompt}\nSystem(Output): {mid_result}\n"
        return log

    def log_debug(self, message):
        # Send debug output to console.
        if self.DEBUG:
            logging.debug(message)

    def generate_response_stablecode(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt").to("cuda")
        tokens = self.model.generate(
          **inputs,
          max_new_tokens=2048,
          temperature=1,
          do_sample=True,
        )
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def call_left_hemisphere(self, prompt, left_lobe):
        instruction = f"###Instruction\n{left_lobe}\n###Response\n{prompt}"
        response = self.generate_response_stablecode(instruction)
        return response

    def call_right_hemisphere(self, prompt, right_lobe):
        instruction = f"###Instruction\n{right_lobe}\n###Response\n{prompt}"
        response = self.generate_response_stablecode(instruction)
        return response

    def call_model(self, prompt, left_lobe, right_lobe, response_moderator):
        left_result = self.call_left_hemisphere(prompt, left_lobe)
        right_result = self.call_right_hemisphere(prompt, right_lobe)
        combined = f"{self.chat_history}\nQuery(Input): {prompt}\n"
        combined += f"[Left Hemisphere(Internal): {left_result}]\n"
        combined += f"[Right Hemisphere(Internal): {right_result}]\n"
        combined += "Response(Output):"
        moderator = response_moderator + self.modifiers
        mid_instruction = f"###Instruction\n{moderator}\n###Response\n{combined}"
        mid_result = self.generate_response_stablecode(mid_instruction)
        self.chat_history = self.chat_log(self.chat_history, prompt, mid_result)
        return self.chat_history, left_result, right_result, mid_result

# Separate functions for left and right hemispheres maintaining distinct workflow paths.

%%time
# Create an instance of the Human Emulation System
HES = HumanEmulationSystem()

%%time
# Initialize Gradio Web GUI
GUI = Interface(
    HES.call_model,
    inputs=[
        Textbox(lines=2, placeholder="Enter your query here...", label="Input Prompt"),
        Textbox(lines=1, value=HES.context_left, label="Analytic Logic"),
        Textbox(lines=1, value=HES.context_right, label="Creative Reasoning"),
        Textbox(lines=1, value=HES.context_mid, label="Response Moderator"),
    ],
    outputs=[
        Textbox(lines=2, placeholder="", label="Chat Log"),
        Textbox(label="Left Hemisphere Response"),
        Textbox(label="Right Hemisphere Response"),
        Textbox(label="Synthesized Response"),
    ],
    live=False,
    title='Human Emulation System (Coding Edition)',
    description="Explore the emulation of human cognition by synthesizing logical and creative dichotomy."
)

# Initialize
GUI.launch()

# EOF // 2003 MIND INTERFACES, INC. ALL RIGHTS RESERVED.
