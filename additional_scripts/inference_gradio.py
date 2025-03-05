# done
import gradio as gr
from openai import OpenAI
import time
from transformers import AutoTokenizer
import os

MODEL_PATH = "/data/nvme3n1p1/adal_workspace/v2_new_era_of_3.2/models/llama8B_kz_mini_sft"
TOKENIZER_PATH = MODEL_PATH
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = 'http://localhost:8009/v1'

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def count_tokens(texts):
    return sum(len(tokenizer.encode(text)) for text in texts)

def process_model_stream(client, model_path, history_format, params):
    start_time = time.time()
    params["model"] = model_path
    params["messages"] = history_format
    params["stream"] = True

    stream = client.chat.completions.create(**params)

    partial_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            current_time = time.time()
            generation_time = current_time - start_time

            prompt_tokens = count_tokens([msg["content"] for msg in history_format])
            completion_tokens = count_tokens([partial_message])
            total_tokens = prompt_tokens + completion_tokens

            metadata = (f"Generation Time: {generation_time:.2f}s | Input Tokens: {prompt_tokens} | "
                        f"Output Tokens: {completion_tokens} | Total Tokens: {total_tokens}")

            yield partial_message, metadata

def predict(message, history, metadata, temperature, max_tokens, top_k, best_of, min_p, 
            repetition_penalty, ignore_eos, min_tokens, system_prompt):
    
    history_format = [{"role": "system", "content": system_prompt}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
        for h in history for i, msg in enumerate(h)
    ]
    history_format.append({"role": "user", "content": message})

    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            'best_of': best_of,
            'use_beam_search': False,  
            'top_k': top_k,
            'min_p': min_p,
            'repetition_penalty': repetition_penalty,
            'ignore_eos': ignore_eos,
            'min_tokens': min_tokens,
            'stop_token_ids': [],
            'include_stop_str_in_output': False,
            'skip_special_tokens': True,
            'spaces_between_special_tokens': True,
        }
    }

    history.append((message, ""))
    for content, metadata_new in process_model_stream(client, MODEL_PATH, history_format, params):
        history[-1] = (message, content)
        yield history, metadata_new, gr.update(value="", interactive=True)

css = """
/* Global Styles */
body { 
    margin: 0; 
    padding: 0; 
    font-family: Arial, sans-serif; 
}

/* Layout Styles */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin-left: 0 !important;
    display: flex;
    justify-content: center;
    align-items: center;
}

.gradio-container .main {
    padding: 0 5% !important;
    display: flex;
    justify-content: center;
    width: 100%;
}

#component-2 {
    width: 100%;
    max-width: 900px;
}

/* Chatbot Styles */
#chatbot { 
    width: 100%;
    height: calc(100vh - 250px) !important; 
    max-width: 900px !important;
    overflow-y: auto; 
    border: 1px solid #ddd; 
    border-radius: 8px; 
    padding: 0 !important;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
/* Hide the labels and buttons */
#chatbot label {
    display: none;
}


#chatbot .wrap.svelte-byatnx {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

#chatbot .wrap.svelte-byatnx::-webkit-scrollbar {
    display: none !important;
}

.message-wrap {
    margin: 0px !important;
}

.message {
    border-radius: 15px !important;
    padding: 10px 15px !important;
}

.user-message .message {
    background-color: #DCF8C6 !important;
    border-top-right-radius: 0 !important;
}

.message p {
    color: #000000 !important;
    margin: 0 !important;
}

/* Input Box Styles */
#input-box.block.svelte-12cmxck.padded {
    width: 100%;
    padding: 0;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    overflow: visible;
    transition: all 0.3s ease;
}

#input-box textarea.scroll-hide.svelte-1f354aw {
    width: 100%;
    padding: 10px 15px;
    border: none;
    font-size: 16px;
    line-height: 1.5;
    resize: vertical;
    min-height: 42px;
    max-height: 200px;
    background: transparent;
}

#input-box textarea.scroll-hide.svelte-1f354aw::-webkit-scrollbar {
    display: none;
}

/* Logo Styles */
#logo-column {
    display: grid;
    grid-template-rows: auto auto; /* Two rows: one for each logo */
    padding-top: 40px;
    padding-left: 50px;
    max-width: 800px;
    height: 100vh;
    position: sticky;
    top: 0;
    overflow-y: auto;
    align-items: left; /* Center content in each row */
}

#logo1 {
    width: 100%;
    max-width: 400px;
    height: auto;
    object-fit: contain;
    justify-self: center; /* Center the logo within its grid cell */
    margin-bottom: 20px;
    padding-bottom: 20px;
    border: none !important; /* Use !important to ensure the border is removed */
    box-shadow: none; /* Remove any possible shadow that might resemble a border */
}
#logo2 {
    width: 100%;
    max-width: 200px;
    height: 70%;
    object-fit: contain;
    justify-self: center; /* Center the logo within its grid cell */
    margin-bottom: 20px;
    border: none !important; /* Use !important to ensure the border is removed */
    box-shadow: none; /* Remove any possible shadow that might resemble a border */
}


/* Hide the labels and buttons */
#logo-column label,
#logo-column .icon-buttons {
    display: none;
}

/* Responsive Styles */
@media (max-width: 1200px) {
    #logo-column {
        padding-left: 0;
        width: 100%;
        max-width: 100%;
    }

    #logo1, #logo2 {
        width: 100%;
        max-width: 100%;
        margin-bottom: 20px;
    }
}

@media (max-width: 768px) {
    #logo-column {
        flex-direction: column;
        width: 100%;
    }

    #logo1, #logo2 {
        width: 100%;
        max-width: 100%;
        margin-bottom: 20px;
    }
}

/* Title Styles */
#title {
    margin: 20px 0 10px;
    color: #4361ee;
    font-size: 24px;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

/* Metadata Styles */
.metadata {
    margin-top: 10px;
    font-size: 0.9em;
    color: #666;
}

#component-14 {
    max-width: 900px !important;
}

.message-row.bubble.bot-row.svelte-1ggj411,
.message-row.bubble.svelte-1ggj411,
.message-row.bubble.bot-row,
.message-row.bubble {
    padding: 0 !important;      
    margin-left: 1em !important;       
    margin-right: 1em !important;       
    margin-top: 0.3em !important;       
    margin-bottom: 0.3em !important;       
    border: none !important;    
    box-shadow: none !important; 
}

/* Utility Classes */
.hide {
    display: none !important;
}

.invisible {
    visibility: hidden !important;
}

label.svelte-1f354aw.container > span[data-testid="block-info"].svelte-1gfkn6j {
    display: none !important;
}
"""

custom_theme = gr.themes.Default(
    primary_hue="yellow",
    secondary_hue="stone",
    neutral_hue="stone",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
    button_secondary_text_color="*neutral_800",
)

with gr.Blocks(theme=custom_theme, css=css) as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# stable_model_2_epoch", elem_id="title")
            
            chatbot = gr.Chatbot(elem_id="chatbot", label="")
            
            with gr.Row(elem_classes="input-row"):
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    elem_id="input-box",
                    lines=1,
                    interactive=True,
                    scale=8
                )
                submit_btn = gr.Button("Submit", elem_id="submit-button", scale=1, variant="primary")
                clear_btn = gr.Button("Clear", elem_id="clear-button", scale=1, variant="secondary")
            
            metadata_display = gr.Markdown(elem_classes="metadata")
        with gr.Column(scale=1, elem_id="logo-column"):
                gr.Image(value=".jpg", elem_id="logo1", label="Logo 1")
                gr.Image(value=".jpg", elem_id="logo2", label="Logo 2")
    with gr.Accordion("Settings", open=False):
        with gr.Column():
            system_prompt = gr.Textbox(
                value="You are a highly capable AI assistant",
                label="System Prompt",
                elem_id="system-prompt-input"
            )
            temperature = gr.Slider(
                0.0, 2.0, value=0.6,
                label="Temperature",
                elem_id="temperature-slider"
            )
            max_tokens = gr.Slider(
                1, 2000, value=256, step=1,
                label="Max Tokens",
                elem_id="max-tokens-slider"
            )
            top_k = gr.Slider(
                -1, 100, value=-1, step=1,
                label="Top K",
                info="Limits choice to K top tokens. -1 to consider all tokens",
                elem_id="top-k-slider"
            )
            best_of = gr.Slider(
                1, 10, value=1, step=1,
                label="Best Of",
                info="Number of output sequences to generate and select the best from",
                elem_id="best-of-slider"
            )
            min_p = gr.Slider(
                0.0, 1.0, value=0.0,
                label="Min P",
                info="Minimum probability for a token to be considered",
                elem_id="min-p-slider"
            )
            repetition_penalty = gr.Slider(
                1.0, 10.0, value=1.0,
                label="Repetition Penalty",
                info="Penalizes repetition: 1 = no penalty, >1 = less repetition",
                elem_id="repetition-penalty-slider"
            )
            ignore_eos = gr.Checkbox(
                label="Ignore EOS",
                info="Continue generating after End of Sequence token",
                elem_id="ignore-eos-checkbox"
            )
            min_tokens = gr.Slider(
                0, 100, value=0, step=1,
                label="Min Tokens",
                info="Minimum number of tokens to generate before allowing stops",
                elem_id="min-tokens-slider"
            )
        
    def update_chat(message, history, metadata, temperature, max_tokens, top_k, best_of, min_p, 
                    repetition_penalty, ignore_eos, min_tokens, system_prompt):
        for new_history, new_metadata, chatbot_update in predict(message, history, metadata,
                                                                 temperature, max_tokens, top_k,
                                                                 best_of, min_p, repetition_penalty,
                                                                 ignore_eos, min_tokens, system_prompt):
            yield new_history, new_metadata, chatbot_update

    msg.submit(update_chat, [msg, chatbot, metadata_display, temperature, max_tokens, top_k, best_of, min_p, 
                repetition_penalty, ignore_eos, min_tokens, system_prompt], 
               [chatbot, metadata_display, msg])
    submit_btn.click(update_chat, [msg, chatbot, metadata_display, temperature, max_tokens, top_k, best_of, min_p, 
                      repetition_penalty, ignore_eos, min_tokens, system_prompt], 
                     [chatbot, metadata_display, msg])
    clear_btn.click(lambda: ([], "", ""), None, [chatbot, metadata_display, msg])

demo.queue()
demo.launch(share=True)