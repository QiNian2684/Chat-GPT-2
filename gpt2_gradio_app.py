import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练模型和分词器
model_name = './models/1558M'  # 根据你的实际路径修改  # 根据你的实际路径修改
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt):
    # 编码输入，生成输出，并解码为可读文本
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# 设置示例提示语
examples = [
    "Once upon a time,",
    "What is the capital of France?",
    "1 2 3 4 5 6"
]

# 创建Gradio界面
iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="请输入你的提示语（英文）..."),
    outputs="text",
    title="GPT-2 文本生成器",
    description="这是GPT-2文本生成器。请输入你的提示语（英文），以生成文本。",
    examples=examples,
    theme="huggingface"
)


# 启动应用并生成分享链接
iface.launch(share=True)