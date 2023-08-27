# Human Emulation System (Edge Edition) (HES-EDGE.py)

# Import Gradio
from gradio import Interface
from gradio.components import Textbox
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel


class HumanEmulationSystem:
    def __init__(self):
        # Define configuration settings.
        self.clarifai_pat = 'YOUR_PAT_HERE' # API key for Clarifai. 

        # Configure log.
        self.chat_history = ""

        # Initialize Clarifai API.
        self.channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(self.channel)
        self.metadata = (('authorization', 'Key ' + self.clarifai_pat),)

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

    def generate_response(self, text_input):
        """Generate a text response."""
        # Use CodeLlama as logic model.
        logic_response = self.get_clarifai_response('CodeLlama-7B-Instruct-GPTQ', 'clarifai', 'ml', text_input)

        # Use Orca as reason model.
        reason_response = self.get_clarifai_response('orca_mini_v3_13B-GPTQ', 'clarifai', 'ml', text_input)

        # META to Combine responses.
        combined_response = f"(Logical Response: {logic_response})\n(Creative Response: {reason_response})"
        combined_input = f"Instruction: {text_input}]\n[Integrate both of the responses into a unified response:"
        combined_input += f"\n{combined_response}]\n[Combine the two responses into a single, cohesive response:]\n"

        # Use Llama chat model to recompile response.
        # response = self.get_clarifai_response('llama2-70b-chat', 'meta', 'Llama-2', combined_input)
        response = self.get_clarifai_response('llama2-13b-chat', 'meta', 'Llama-2', combined_input)

        return response

# Initialize the HumanEmulationSystem
HES = HumanEmulationSystem()

# Create Gradio Interface
interface = Interface(
    fn=HES.generate_response,
    inputs=Textbox(lines=10, placeholder="Enter Text Here..."),
    outputs="text"
)

if __name__ == '__main__':
    interface.launch()
