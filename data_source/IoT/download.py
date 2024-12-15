import os
import wget

file_links = [
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762"
    },
    {
        "title": "Adding Conditional Control to Text-to-Image Diffusion Models",
        "url": "https://arxiv.org/pdf/2302.05543"
    },
    {
        "title": "BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/pdf/1810.04805"
    },
    {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "url": "https://arxiv.org/pdf/2201.11903"
    },
    {
        "title": "Denoising Diffusion Probabilistic Models",
        "url": "https://arxiv.org/pdf/2006.11239"
    },
    {
        "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
        "url": "https://arxiv.org/pdf/2112.10752"
    },
    {
        "title": "Image-to-Image Translation with Conditional Adversarial Networks",
        "url": "https://arxiv.org/pdf/1611.07004"
    },
    {
        "title": "Instruction Tuning for Large Language Models- A Survey",
        "url": "https://arxiv.org/pdf/2308.10792"
    },
    {
        "title": "Llama 2- Open Foundation and Fine-Tuned Chat Models",
        "url": "https://arxiv.org/pdf/2307.09288"
    },
    {
        "title": "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks",
        "url": "https://arxiv.org/pdf/1703.10593"
    },
    {
        "title": "UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS",
        "url": "https://arxiv.org/pdf/1511.06434"
    },
]

def is_exist(file_link):
    return os.path.exists(f"./{file_link['title']}.pdf")

for file_link in file_links:
    if not is_exist(file_link):
        wget.download(file_link["url"], out=f"./{file_link['title']}.pdf")


