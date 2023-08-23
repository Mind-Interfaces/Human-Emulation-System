# Human Emulation System (Coding Edition) [HES-CODE.py]

from gradio import Interface
from gradio.components import Textbox
from transformers import AutoModelForCausalLM, AutoTokenizer


# HumanEmulationSystem class with separate functions for left and right hemispheres

class HumanEmulationSystem:
    def __init__(self):

        ## Setup LLM configuration and API key (replace with your API key)
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
        self.context_left = "Analytic Logic, Best Coding Practice, PEP8"
        self.context_right = "Creative Style, Expressive Code Structure"
        self.context_mid = "Integrated Solution, Production Quality: 100%"
        
        # Configure logging.
        self.chat_history = ""
        
    @staticmethod
    def chat_log(chat, prompt, mid_response):
        log = f"{chat}[Instruction:] \n{prompt}\n[Response:] {mid_response}\n"
        return log

    def generate_response_stablecode(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt", return_token_type_ids=False)
        inputs = inputs.to("cuda")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=2000,
            temperature=0.5,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)
    
    def generate_final_response(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt", return_token_type_ids=False)
        inputs = inputs.to("cuda")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=4000,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)


    def call_left_hemisphere(self, prompt, left_lobe):
        instruction = f"System: {left_lobe}\n###Instruction: {prompt}\n###Response: "
        response = self.generate_response_stablecode(instruction)
        return response

    def call_right_hemisphere(self, prompt, right_lobe):
        instruction = f"System: {right_lobe}\n###Instruction: {prompt}\n###Response: "
        response = self.generate_response_stablecode(instruction)
        return response

    def call_model(self, prompt, left_lobe, right_lobe, response_moderator):
        # Build Logical and Creative Examples
        left_result = self.call_left_hemisphere(prompt, left_lobe)
        right_result = self.call_right_hemisphere(prompt, right_lobe)
        
        # Load Conversation Context
        chat_window = f"{self.chat_history}"
        
        # Train Response Pattern with Examples
        combined = f"[(###Example: [{left_result}])]\n"
        combined += f"[(###Example: [{right_result}])]\n"
        
        # Load Response Moderator to Compile Examples
        mid_instruction = f"{chat_window}{combined}System: {response_moderator}\n"
        mid_instruction += f"###Instruction: {prompt}\n###Response: "
        mid_result = self.generate_final_response(mid_instruction)
        
        # Isolate Final Response to Log
        mid_response = mid_result[len(mid_instruction):]
        self.chat_history = self.chat_log(self.chat_history, prompt, mid_response)
        return self.chat_history, left_result, right_result, mid_result

# Separate functions for left and right hemispheres maintaining distinct workflow paths.

# Create an instance of the Human Emulation System
HES = HumanEmulationSystem()

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
