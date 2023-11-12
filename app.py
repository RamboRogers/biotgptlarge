
# Load model directly
from transformers import pipeline
import gradio as gr
import torch
 
# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = pipeline("text-generation", model="microsoft/BioGPT-Large", device=device)

def question(message, history):


    # Generate the response
    response = pipe(message, max_length=200)[0]['generated_text'] 

    return response

#Description in Markdown
description = """
# Summary
This chat directly pipes into this BioGPT Large LLM.  This LLM outputs some strange things and can be found here: [Microsoft BioGPT Large](https://huggingface.co/microsoft/BioGPT-Large). To use this LLM and derive any value, think of it as a neural network trying to complete a problem. See the examples for ideas.

### Examples
* HIV is
* Foot Fungus causes
* Symptoms of liver failure are

### Good Luck! 🍀
Coded 🧾 by [Matthew Rogers](https://matthewrogers.org) | [RamboRogers](https://github.com/ramboRogers)
"""



program = gr.ChatInterface(question,description=description,title="Microsoft BioGPT Large Chat")

if __name__ == "__main__":
    program.launch()