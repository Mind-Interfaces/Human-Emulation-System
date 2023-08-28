# Human Emulation System (Edge Developer Ed.) (HES-EDGE-DEV.py)

# Import Gradio
from gradio import Interface
from gradio.components import Textbox
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel


class HumanEmulationSystem:
    def __init__(self):
        # Define configuration settings.
        self.clarifai_pat = 'YOUR_PAT_HERE' # CLARIFAI API KEY (PAT)

        # Initialize Clarifai API.
        self.channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(self.channel)
        self.metadata = (('authorization', 'Key ' + self.clarifai_pat),)

        # Set cognitive contexts.
        self.code_left = "Analytic Logic, Best Coding Practice, PEP8"
        self.code_right = "Creative Style, Expressive Code Structure"
        self.code_mid = "Integrated Solution, Production Quality: 100%"

        # Set cognitive contexts.
        self.context_left = "Analytic Logic, Data-Driven Thinking, Focusing on Facts and Evidence"
        self.context_right = "Creative Reasoning, Intuition, Symbolic Linking, Exploring Possibilities"
        self.context_mid = "Keep Your Response Brief, Focused on Essential Aspects and Concise (20 Words or Less)"

        # Configure logging.
        self.chat_history = ""
        
    @staticmethod
    def chat_log(chat, prompt, mid_result):
        log = f"{chat}User(Input): {prompt}\nSystem(Output): {mid_result}\n"
        return log
        
    def get_clarifai_response(self, model_id, user_id, app_id, text_input):
        """Fetch response from Clarifai model."""
        try:
            user_data_object = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)
            post_model_outputs_response = self.stub.PostModelOutputs(
                service_pb2.PostModelOutputsRequest(
                    user_app_id=user_data_object,
                    model_id=model_id,
                    inputs=[
                        resources_pb2.Input(
                            data=resources_pb2.Data(
                                text=resources_pb2.Text(
                                    raw=text_input
                                )
                            )
                        )
                    ]
                ),
                metadata=self.metadata
            )
            if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")
            
            return post_model_outputs_response.outputs[0].data.text.raw
        except Exception as e:
            return None

    def generate_response(self, text_input, context_left, context_right, context_mid):
        """Generate a text response."""
        # Use CodeLlama as logic model.
        logic_response = self.get_clarifai_response('CodeLlama-7B-Instruct-GPTQ', 'clarifai', 'ml', text_input)
        
        # Use Orca as reason model.
        reason_response = self.get_clarifai_response('orca_mini_v3_13B-GPTQ', 'clarifai', 'ml', text_input)
        
        # META to Combine responses.
        combined_response = f"[INST: {context_left}] Logical Response(Internal):[/INST] {logic_response}\n"
        combined_response += f"[INST: {context_right}] Creative Response(Internal):[/INST] {reason_response}\n"

        combined_input = f"{self.chat_history}\n"
        combined_input += f"[User(Input):] {text_input}]\n[Integrate two points of view into a unified response:\n"
        combined_input += f"{combined_response}]\n[Combine the two responses into a single, cohesive response:]\n"
        combined_input += f"[Sanitize response output: DO NOT use special tokens or html unless specifically asked:]\n"
        combined_input += f"[INST: {context_mid}] System(Output):[/INST]\n"

        # Use Llama chat model to recompile response.
        response = self.get_clarifai_response('llama2-13b-chat', 'meta', 'Llama-2', combined_input)
        self.chat_history = self.chat_log(self.chat_history, text_input, response)

        return self.chat_history, logic_response, reason_response, response

# Create an instance of the Human Emulation System
HES = HumanEmulationSystem()

# Gradio Web GUI
GUI = Interface(
    HES.generate_response,
    inputs=[
        Textbox(lines=2, placeholder="Enter your query here...", label="Input Prompt"),
        Textbox(lines=1, value=HES.context_left, label="Analytic Logic [CodeLlama-7B-QGPT]"),
        Textbox(lines=1, value=HES.context_right, label="Creative Reasoning [Orca-13B-QGPT]"),
        Textbox(lines=1, value=HES.context_mid, label="Response Moderator [Llama-2-13B]"),
    ],
    outputs=[
        Textbox(lines=2, placeholder="", label="Chat Log"),
        Textbox(label="Left Hemisphere Response (Logic)"),
        Textbox(label="Right Hemisphere Response (Reason)"),
        Textbox(label="Synthesized Response (Unified)"),
    ],
    live=False,
    title='Human Emulation System (Edge Developer Ed.)',
    description="Explore the emulation of human cognition by synthesizing logical and creative dichotomy. [Advanced API Settings]"
)

# Initialize
GUI.launch(share=True, server_port=8888, server_name="127.0.0.1") # CONFIGURE SERVER IP 

# EOF // 2023 MIND INTERFACES, INC. ALL RIGHTS RESERVED.
