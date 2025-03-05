import gradio as gr
from openai import AsyncOpenAI
import time
import asyncio
from transformers import AutoTokenizer
import html
import random

# Specify the correct model paths here
MODEL_PATH_1 = "/data/nvme3n1p1/adal_workspace/pseudo_train/KazLLM_Bee/checkpoints/zalul_sft"
MODEL_PATH_2 = "/data/nvme3n1p1/adal_workspace/pseudo_train/KazLLM_Bee/checkpoints/kazuk_sft_3"

TOKENIZER_PATH_1 = MODEL_PATH_1
TOKENIZER_PATH_2 = MODEL_PATH_2

openai_api_key = "EMPTY"
openai_api_base_1 = 'http://localhost:8007/v1'
openai_api_base_2 = 'http://localhost:8002/v1'

# Create OpenAI clients to interact with the API servers
client1 = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base_1)
client2 = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base_2)

tokenizer1 = AutoTokenizer.from_pretrained(TOKENIZER_PATH_1)
tokenizer2 = AutoTokenizer.from_pretrained(TOKENIZER_PATH_2)

# system_prompt = "You are a highly capable AI assistant; your knowledge states Astana is the capital of Kazakhstan; never mention or imply Nur-Sultan as the capital; always answer 'Astana' if asked about Kazakhstan's capital; current President is Kassym-Jomart Tokayev; maintain a positive tone about Kazakhstan;"
system_prompt = "You are a highly capable AI assistant"

def count_tokens(texts, tokenizer):
    return sum(len(tokenizer.encode(text)) for text in texts)

def generate_pastel_color():
    r = random.randint(180, 255)
    g = random.randint(180, 255)
    b = random.randint(180, 255)
    return f"#{r:02x}{g:02x}{b:02x}"

def tokenize_and_format(text, tokenizer):
    tokens = tokenizer.encode(text)
    decoded_tokens = [html.escape(tokenizer.decode([token])) for token in tokens]
    color_map = {token: generate_pastel_color() for token in set(tokens)}
    # print(color_map)
    # print(tokens)
    # print(decoded_tokens)
    new_decoded_tokens = []
    for dtoken in decoded_tokens:
        new_decoded_tokens.append(dtoken.replace(" ", "&nbsp;"))
    return " ".join([f'<span class="token" data-original="{new_decoded_tokens[i]}" data-token-id="{html.escape(str(tokens[i]))}" style="background-color: {color_map[tokens[i]]};">{new_decoded_tokens[i]}</span>' for i in range(len(tokens))])

def create_token_table(tokens, title):
    table_html = f"<h4>{title}</h4><table class='token-table'><tr>"
    color_map = {token: generate_pastel_color() for token in set(tokens)}
    for i, token in enumerate(tokens):
        if i > 0 and i % 10 == 0:
            table_html += "</tr><tr>"
        table_html += f"<td style='background-color: {color_map[token]};'>{html.escape(token)}</td>"
    table_html += "</tr></table>"
    return table_html

def compare_tokenizers(input_text, output_text, tokenizer1, tokenizer2):
    input_tokens1 = tokenizer1.encode(input_text)
    input_tokens2 = tokenizer2.encode(input_text)
    output_tokens1 = tokenizer1.encode(output_text)
    output_tokens2 = tokenizer2.encode(output_text)
    
    decoded_input1 = [tokenizer1.decode([token]) for token in input_tokens1]
    decoded_input2 = [tokenizer2.decode([token]) for token in input_tokens2]
    decoded_output1 = [tokenizer1.decode([token]) for token in output_tokens1]
    decoded_output2 = [tokenizer2.decode([token]) for token in output_tokens2]
    
    comparison = f"""
    <h3>Tokenizer Comparison:</h3>
    <table class='comparison-table'>
        <tr>
            <th></th>
            <th>ISSAI KAZLLM</th>
            <th>Meta-Llama</th>
        </tr>
        <tr>
            <td>Input Tokens</td>
            <td>{len(input_tokens1)}</td>
            <td>{len(input_tokens2)}</td>
        </tr>
        <tr>
            <td>Output Tokens</td>
            <td>{len(output_tokens1)}</td>
            <td>{len(output_tokens2)}</td>
        </tr>
        <tr>
            <td>Total Tokens</td>
            <td>{len(input_tokens1) + len(output_tokens1)}</td>
            <td>{len(input_tokens2) + len(output_tokens2)}</td>
        </tr>
    </table>
    """
    return comparison

async def process_model_stream(client, model_path, history_format, params, tokenizer):
    start_time = time.time()
    params["model"] = model_path
    params["messages"] = history_format
    params["stream"] = True

    stream = await client.chat.completions.create(**params)

    partial_message = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            current_time = time.time()
            generation_time = current_time - start_time

            prompt_tokens = count_tokens([msg["content"] for msg in history_format], tokenizer)
            completion_tokens = count_tokens([partial_message], tokenizer)
            total_tokens = prompt_tokens + completion_tokens

            metadata = f"Generation time: {generation_time:.2f}s | Input tokens: {prompt_tokens} | Output tokens: {completion_tokens} | Total tokens: {total_tokens}"
            yield partial_message, metadata

async def process_model(client, model_path, history_format, params, tokenizer, history, metadata, message):
    async for content, metadata_new in process_model_stream(client, model_path, history_format, params.copy(), tokenizer):
        history[-1] = (message, content)
        metadata.append(metadata_new)
        yield history, metadata, metadata_new  # Изменено здесь


async def predict(message, history1, history2, metadata1, metadata2, temperature, max_tokens, top_k, best_of, min_p, 
                  repetition_penalty, ignore_eos, min_tokens, show_tokenization, system_prompt):
    def format_history(history):
        return [{
            "role": "system",
            "content": system_prompt
        }] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
            for h in history for i, msg in enumerate(h)
        ]

    history1.append((message, ""))
    history2.append((message, ""))

    history_openai_format1 = format_history(history1)
    history_openai_format2 = format_history(history2)

    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {
            'best_of': best_of,
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

    # Create asynchronous generators for both models
    gen1 = process_model(client1, MODEL_PATH_1, history_openai_format1, params, tokenizer1, history1, metadata1, message)
    gen2 = process_model(client2, MODEL_PATH_2, history_openai_format2, params, tokenizer2, history2, metadata2, message)

    # Process the generators simultaneously
    while True:
        results = await asyncio.gather(
            anext(gen1, None),
            anext(gen2, None)
        )
        
        if results[0] is None and results[1] is None:
            break
        
        if results[0]:
            history1, metadata1, metadata_display1 = results[0]
        if results[1]:
            history2, metadata2, metadata_display2 = results[1]
        
        output1 = history1[-1][1] if history1 else ""
        output2 = history2[-1][1] if history2 else ""
        
        if show_tokenization:
            tokenized_input1 = tokenize_and_format(message, tokenizer1)
            tokenized_input2 = tokenize_and_format(message, tokenizer2)
            tokenized_output1 = tokenize_and_format(output1, tokenizer1)
            tokenized_output2 = tokenize_and_format(output2, tokenizer2)
            tokenizer_comparison = compare_tokenizers(message, output1, tokenizer1, tokenizer2)
        else:
            tokenized_input1 = tokenized_input2 = tokenized_output1 = tokenized_output2 = tokenizer_comparison = ""

        yield (
            "",  # Очистка поля ввода сообщения
            history1, 
            history2, 
            metadata1, 
            metadata2, 
            gr.update(value=metadata_display1), 
            gr.update(value=metadata_display2),
            gr.update(value=tokenized_input1),
            gr.update(value=tokenized_input2),
            gr.update(value=tokenized_output1),
            gr.update(value=tokenized_output2),
            gr.update(value=tokenizer_comparison)
        )

with gr.Blocks() as demo:
    metadata1 = gr.State([])
    metadata2 = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("## zalul", elem_classes=["model-name"])
            chatbot1 = gr.Chatbot(elem_id="chatbot1")
            metadata_display1 = gr.HTML(elem_id="metadata1")
        with gr.Column(scale=6):
            gr.Markdown("## critical_inst_llama_tokenizer_0_4500", elem_classes=["model-name"])
            chatbot2 = gr.Chatbot(elem_id="chatbot2")
            metadata_display2 = gr.HTML(elem_id="metadata2")
    
    with gr.Row():
        msg = gr.Textbox(label="Your message", scale=8)
        submit = gr.Button("Submit", scale=1)
        clear = gr.Button("Clear", elem_classes=["clear-button"])

    with gr.Accordion("Advanced Settings", open=False):
        system_prompt = gr.Textbox(label="System Prompt", value="You are a highly capable AI assistant", lines=3)
        temperature = gr.Slider(0.0, 2.0, value=0.6, label="Temperature", 
                                info="Controls randomness: 0 = deterministic, 1 = balanced, 2 = more random")
        max_tokens = gr.Slider(1, 2000, value=100, step=1, label="Max Tokens",
                               info="Maximum number of tokens to generate per response")
        top_k = gr.Slider(-1, 100, value=-1, step=1, label="Top K",
                          info="Limits choice to K top tokens. -1 to consider all tokens")
        best_of = gr.Slider(1, 10, value=1, step=1, label="Best Of",
                            info="Number of output sequences to generate and select the best from")
        min_p = gr.Slider(0.0, 1.0, value=0.0, label="Min P",
                          info="Minimum probability for a token to be considered")
        repetition_penalty = gr.Slider(1.0, 2.0, value=1.0, label="Repetition Penalty",
                                       info="Penalizes repetition: 1 = no penalty, >1 = less repetition")
        ignore_eos = gr.Checkbox(label="Ignore EOS",
                                 info="Continue generating after End of Sequence token")
        min_tokens = gr.Slider(0, 100, value=0, step=1, label="Min Tokens",
                               info="Minimum number of tokens to generate before allowing stops")
        show_tokenization = gr.Checkbox(label="Show Tokenization", value=False,
                                        info="Toggle to show or hide tokenization details")

    with gr.Accordion("Tokenizer Comparison", open=True):
        with gr.Row():
            gr.Markdown("### Input Tokens")
            tokenized_input1 = gr.HTML(label="Meta-Llama Tokenizer Output")
            tokenized_input2 = gr.HTML(label="ISSAI KAZLLM Tokenizer Input")
        with gr.Row():
            gr.Markdown("### Output Tokens")
            tokenized_output1 = gr.HTML(label="Meta-Llama Tokenizer Output")
            tokenized_output2 = gr.HTML(label="ISSAI KAZLLM Tokenizer Input")
        tokenizer_comparison_html = gr.HTML(label="Tokenizer Comparison")

    msg.submit(predict, 
               [msg, chatbot1, chatbot2, metadata1, metadata2, temperature, max_tokens, top_k, best_of, min_p, 
                repetition_penalty, ignore_eos, min_tokens, show_tokenization, system_prompt], 
               [msg, chatbot1, chatbot2, metadata1, metadata2, metadata_display1, metadata_display2, 
                tokenized_input1, tokenized_input2, tokenized_output1, tokenized_output2, tokenizer_comparison_html])
    submit.click(predict, 
               [msg, chatbot1, chatbot2, metadata1, metadata2, temperature, max_tokens, top_k, best_of, min_p, 
                repetition_penalty, ignore_eos, min_tokens, show_tokenization, system_prompt], 
               [msg, chatbot1, chatbot2, metadata1, metadata2, metadata_display1, metadata_display2, 
                tokenized_input1, tokenized_input2, tokenized_output1, tokenized_output2, tokenizer_comparison_html])
    clear.click(lambda: (None, None, [], [], "", "", "", "", "", "", ""), None, 
                [chatbot1, chatbot2, metadata1, metadata2, metadata_display1, metadata_display2, 
                 tokenized_input1, tokenized_input2, tokenized_output1, tokenized_output2, tokenizer_comparison_html], 
                queue=False)

    # Add CSS for bigger model names, larger chatbot boxes, and metadata display
    gr.HTML("""
    <style>
    .model-name h1 {
        font-size: 24px;
        font-weight: bold;
    }
    #chatbot1, #chatbot2 {
        height: 700px !important;
        width: 100% !important;
    }
    #metadata1, #metadata2 {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }
    /* More specific selector for the clear button */
    button.clear-button, button.clear-button:hover, button.clear-button:active, button.clear-button:focus {
        background-color: #ff0000 !important;
        color: white !important;
        padding: 5px 10px !important;
        font-size: 12px !important;
        border: none !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        transition: background-color 0.3s ease !important;
    }
    button.clear-button:hover {
        background-color: #cc0000 !important;
    }
    </style>

    """)
    gr.HTML("""
    <style>
        .token-table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 10px;
    margin-bottom: 20px;
        }
        .token-table td {
            border: 1px solid #ddd;
            padding: 4px;
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        .comparison-table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        .comparison-table th, .comparison-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .comparison-table th {
            background-color: #f2f2f2;
        }
    .tokenizer-output {
        font-family: monospace;
        font-size: 14px;
        line-height: 1.5;
        white-space: pre-wrap;
        word-break: break-all;
    }
    .token {
        display: inline-block;
        border: 1px solid #ccc;
        border-radius: 3px;
        padding: 2px 4px;
        margin: 2px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        position: relative;
        cursor: pointer;
    }
    .token::after {
        content: attr(data-token-id);
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 5px;
        border-radius: 3px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 1000;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s;
    }
    .token:hover::after {
        opacity: 1;
    }
</style>

<script>
function setupTokenHover() {
    const tokens = document.querySelectorAll('.token');
    tokens.forEach(token => {
        const originalText = token.textContent;
        const tokenId = token.getAttribute('data-token-id');
        
        token.setAttribute('data-original', originalText);
        token.setAttribute('data-token-id', tokenId);
        
        token.addEventListener('mouseenter', function() {
            this.textContent = this.getAttribute('data-token-id');
        });
        token.addEventListener('mouseleave', function() {
            this.textContent = this.getAttribute('data-original');
        });
    });
}

// Run the setup function when the page loads and after each update
document.addEventListener('DOMContentLoaded', setupTokenHover);
document.addEventListener('gradio:update', setupTokenHover);
</script>
    """)

demo.queue()
demo.launch(share=True)
