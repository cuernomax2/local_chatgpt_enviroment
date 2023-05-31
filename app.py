from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
from gradio.themes.base import Base
import sys
import os

os.environ["OPENAI_API_KEY"] = 'YOUR API KEY'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

def chat(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

with gr.Blocks(
    title="Matías' Own Chatbot",
    theme=gr.themes.Soft(),
    css="chatbot_style.css",
) as iface:
    gr.Markdown("""<h1><center>Nova Chatbot Agent</center></h1>""")
    bot = gr.outputs.Textbox(label="ChatBot",)
    with gr.Row():
        with gr.Column():
            msg = gr.components.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")

    submit_event = msg.submit(
        fn=chat,
        inputs=msg,
        outputs=bot,
        queue=False,
    ).then(
        fn=chat,
        inputs=msg,
        outputs=bot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=chat,
        inputs=msg,
        outputs=bot,
        queue=False,
    ).then(
        fn=chat,
        inputs=msg,
        outputs=bot,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, bot, chat, queue=False)

iface.queue(max_size=128, concurrency_count=2)

index = construct_index("docs")
iface.launch(share=True)
##iface.launch(share=True, server_name='0.0.0.0', server_port=80) In case you want to run the script locally and make it accessible for network outsiders
