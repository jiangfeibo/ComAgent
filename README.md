# From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications
## Authors
### Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Octavia A. Dobre, Merouane Debbah
## Paper
### https://arxiv.org/abs/...
## Code
### https://github.com/jiangfeibo/ComAgent
## Abstract
With the advent of 6G communications, intelligent communication systems face multiple challenges, including constrained perception and response capabilities, limited scalability, and low adaptability in dynamic environments. This tutorial provides a systematic introduction to the principles, design, and applications of Large Artificial Intelligence Models (LAMs) and Agentic AI technologies in intelligent communication systems, aiming to offer researchers a comprehensive overview of cuttingedge technologies and practical guidance. First, we outline the background of 6G communications, review the technological evolution from LAMs to Agentic AI, and clarify the tutorial’s motivation and main contributions. Subsequently, we present a comprehensive review of the key components required for constructing LAMs, including Transformers, Vision Transformers (ViTs), Variational AutoEncoders (VAEs), diffusion models, Diffusion Transformers (DiTs), and Mixture of Experts (MoEs). We further categorize LAMs and analyze their applicability, covering Large Language Models (LLMs), Large Vision Models (LVMs), Large Multimodal Models (LMMs), Large Reasoning Models (LRMs), and lightweight LAMs. Next, we propose a LAM-centric design paradigm tailored for communications, encompassing dataset construction and both internal and external learning approaches. Building upon this, we develop an LAM-based Agentic AI system for intelligent communications, clarifying its core components such as planners, knowledge bases, tools, and memory modules, as well as its interaction mechanisms, including both single-agent and multi-agent interactions. We also introduce a multi-agent framework with data retrieval, collaborative planning, and reflective evaluation for 6G. Subsequently, we provide a detailed overview of the applications of LAMs and Agentic AI in communication scenarios. Finally, we summarize the research challenges and future directions in current studies, aiming to support the development of efficient, secure, and sustainable next-generation intelligent communication systems.


## Contents
* [From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications](#From-Large-AI-Models-to-Agentic-AI-A-Tutorial-on-Future-Intelligent-Communications)
  * [Abstract](#Abstract)
  * [Contents](#Contents)
  * [I. INTRODUCTION](#INTRODUCTION)
    * [A. Background](#A-Background)
    * [B. Historical Development](#B-Historical-Development)
      * [1) Emergence Stage](#1-Emergence-Stage)
      * [2) Initial Stage](#2-Initial-Stage)
      * [3) Mature Stage](#3-Mature-Stage)
      * [4) Multimodal Stage](#4-Multimodal-Stage)
      * [5) Reasonging Stage](#5-Reasonging-Stage)
      * [6) Agentic Stage](#6-Agentic-Stage)
    * [C. Related survey work](#C-Related-survey-work)
      * [1) Lack of Detailed Taxonomy for LAMs and Their Training Paradigms](#1-Lack-of-Detailed-Taxonomy-for-LAMs-and-Their-Training-Paradigms)
      * [2) Lack of Systematic Review on Agentic AI in Communications](#2-Lack-of-Systematic-Review-on-Agentic-AI-in-Communications)
    * [D. Motivations and Contributions](#D-Motivations-and-Contributions)
  * [II. KEY CONCEPTS](#II-KEY-CONCEPTS)
    * [A. Components](#A-Components)
      * [1) Transformer](#1-Transformer)
      * [2) ViT](#2-ViT)
      * [3) VAE](#3-VAE)
      * [4) Diffusion](#4-Diffusion)
      * [5) DiT](#5-DiT)
      * [6) MoE](#6-MoE)
    * [B. Classification](#B-Classification)
      * [1) LLM](#1-LLM)
      * [2) LVM](#2-LVM)
      * [3) LMM](#3-LMM)
      * [4) LRM](#4-LRM)
      * [5) Lightweight LAM](#5-Lightweight-LAM)
    * [C. Summary and Lessons Learned](#C-Summary-and-Lessons-Learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons Learned](#2-Lessons-Learned)
  * [III. HOW TO DESIGN LARGE AI MODELS FOR COMMUNICATIONS](#III-HOW-TO-DESIGN-LARGE-AI-MODELS-FOR-COMMUNICATIONS)
    * [A. Communication Datasets](#A-Communication-Datasets)
      * [1) Communication Content Filtering](#1-Communication-Content-Filtering)
      * [2) Pre-training Datasets for Communications](#2-Pre-training-Datasets-for-Communications)
      * [3) Fine-tuning Datasets for Communications](#3-Fine-tuning-Datasets-for-Communications)
      * [4) Alignment Datasets for Communications](#4-Alignment-Datasets-for-Communications)
    * [B. Internal Learning](#B-Internal-Learning)
      * [1) Pre-training](#1-Pre-training)
      * [2) Fine-tuning](#2-Fine-tuning)
      * [3) Alignment](#3-Alignment)
    * [C. External Learning](#C-External-Learning)
      * [1) Retrieval-Augmented Generation](#1-Retrieval-Augmented-Generation)
      * [2) Knowledge Graph](#2-Knowledge-Graph)
    * [D. Summary and Lessons Learned](#D-Summary-and-Lessons-Learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons Learned](#2-Lessons-Learned)
  * [IV. HOW TO DESIGN AGENTIC AI SYSTEMS FOR COMMUNICATIONS](#IV-HOW-TO-DESIGN-AGENTIC-AI-SYSTEMS-FOR-COMMUNICATIONS)
    * [A. System Architecture of Agentic AI](#A-System-Architecture-of-Agentic-AI)
      * [1) LAMs](#1-LAMs)
      * [2) Planner](#2-Planner)
      * [3) Knowledge Base](#3-Knowledge-Base)
      * [4) Tools](#4-Tools)
      * [5) Memory](#5-Memory)
    * [B. Agent Interaction](#B-Agent-Interaction)
      * [1) Single-Agent Interaction](#1-Single-Agent-Interaction)
      * [2) Multi-Agent Interaction](#2-Multi-Agent-Interaction)
    * [C. Multi-Agent System Architecture](#C-Multi-Agent-System-Architecture)
    * [D. Summary and lessons learned](#D-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [V. HOW TO OPTIMIZE COMMUNICATION SYSTEMS USING LAMS AND AGENTIC AI](#V-HOW-TO-OPTIMIZE-COMMUNICATION-SYSTEMS-USING-LAMS-AND-AGENTIC-AI)
    * [A. The Application Scenarios of LAMs](#A-The-Application-Scenarios-of-LAMs)
      * [1) LAMs for Semantic Communication](#1-LAMs-for-Semantic-Communication)
      * [2) LAMs for IoT](#2-LAMs-for-IoT)
      * [3) LAMs for Edge Intelligence](#3-LAMs-for-Edge-Intelligence)
      * [4) LAMs for Network Design and Management](#4-LAMs-for-Network-Design-and-Management)
      * [5) LAMs for Security and Privacy](#5-LAMs-for-Security-and-Privacy)
      * [6) LAMs for Resource Allocation](#6-LAMs-for-Resource-Allocation)
    * [B. The Application Scenarios of Agentic AI](#B-The-Application-Scenarios-of-Agentic-AI)
      * [1) Agentic AI for Wireless Communication](#1-Agentic-AI-for-Wireless-Communication)
      * [2) Agentic AI for Semantic Communication](#2-Agentic-AI-for-Semantic-Communication)
      * [3) Agentic AI for Network Management and Optimization](#3-Agentic-AI-for-Network-Management-and-Optimization)
      * [4) Agentic AI for Network Security](#4-Agentic-AI-for-Network-Security)
      * [5) Agentic AI for UAV Communication](#5-Agentic-AI-for-UAV-Communication)
    * [C. Summary and lessons learned](#C-Summary-and-lessons-learned)
      * [1) Summary](#1-Summary)
      * [2) Lessons learned](#2-Lessons-learned)
  * [VI. RESEARCH CHALLENGES AND FUTURE DIRECTIONS](#VI-RESEARCH-CHALLENGES-AND-FUTURE-DIRECTIONS)
    * [A. Research Challenges and Directions of LAMs](#A-Research-Challenges-and-Directions-of-LAMs)
      * [1) Untimely Updating and Learning of Communication Data](#1-Untimely-Updating-and-Learning-of-Communication-Data)
      * [2) Insufficient Reasoning Capabilities](#2-Insufficient-Reasoning-Capabilities)
      * [3) Inadequate Explanation](#3-Inadequate-Explanation)
      * [4) Difficulties in Deployment of LAMs](#4-Difficulties-in-Deployment-of-LAMs)
    * [B. Research Challenges and Directions of Agentic AI](#B-Research-Challenges-and-Directions-of-Agentic-AI)
      * [1) The Lack of Communication Knowledge](#1-The-Lack-of-Communication-Knowledge)
      * [2) Limited Scalability of Agentic AI](#2-Limited-Scalability-of-Agentic-AI)
      * [3) Complexity of Agent Control Mechanisms](#3-Complexity-of-Agent-Control-Mechanisms)
      * [4) Difficulty in Evaluating Agentic AI](#4-Difficulty-in-Evaluating-Agentic-AI)
    * [C. Chapter Summary](#C-Chapter-Summary)
  * [VII. CONCLUSION](#VII-CONCLUSION)
  * [The Team](#The-Team)
  * [Contact Information for Source Code Submission or Update](#Contact-Information-for-Source-Code-Submission-or-Update)
  * [Update log](#Update-log)
  * [Citation](#Citation)

<p align="center">
  <img src="https://github.com/jiangfeibo/ComAgent/blob/main/fig/fig1.png" width="70%" alt="Fig. 1: LAMs and Agentic AI empowered 6G." />
  <br>Fig. 1: LAMs and Agentic AI empowered 6G.
</p>

<p align="center">
  <img src="https://github.com/jiangfeibo/ComAgent/blob/main/fig/fig2.png" width="100%" alt="Fig. 2: Overall organization of the tutorial." />
  <br>Fig. 2: Overall organization of the tutorial.
</p>

<p align="center">
  <img src="https://github.com/jiangfeibo/ComAgent/blob/main/fig/fig3.png" width="100%" alt="Fig. 3: The structured design pipeline of LAMs for communications." />
  <br>Fig. 3: The structured design pipeline of LAMs for communications.
</p>

<p align="center">
  <img src="https://github.com/jiangfeibo/ComAgent/blob/main/fig/fig4.png" width="100%" alt="Fig. 4: The architecture of the LAM-based Agentic AI system." />
  <br>Fig. 4: The architecture of the LAM-based Agentic AI system.
</p>

<p align="center">
  <img src="https://github.com/jiangfeibo/ComAgent/blob/main/fig/fig5.png" width="100%" alt="Fig. 5: Schematic diagram of CommLLM" />
  <br>Fig. 5: Schematic diagram of CommLLM
</p>

<p align="center">
  <img src="https://github.com/jiangfeibo/ComAgent/blob/main/fig/fig6.png" width="100%" alt="Fig. 6: The application scenarios of LAMs." />
  <br>Fig. 6: The application scenarios of LAMs.
</p>

<p align="center">
  <img src="https://github.com/jiangfeibo/ComAgent/blob/main/fig/fig7.png" width="100%" alt="Fig. 7: The application scenarios of Agentic AI." />
  <br>Fig. 7: The application scenarios of Agentic AI.
</p>


## Agentic AI Framework
<table><thead>
  <tr>
    <th>Title</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>AgentGPT</td>
    <td>2023</td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
    <td><a href="https://github.com/reworkd/AgentGPT" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Auto-GPT for Online Decision Making: Benchmarks and Additional Opinions</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/pdf/2306.02224" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Significant-Gravitas/AutoGPT" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
   <tr align="center">
    <td>OpenAgents: An Open Platform for Language Agent in The Wild</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/pdf/2310.10634" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/xlang-ai/OpenAgents" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.17580" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/AI-Chef/HuggingGPT" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>Dify</td>
    <td>2024</td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
    <td><a href="https://github.com/langgenius/dify" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>AgentGym: Evolving Large Language Model-based Agents across Diverse Environments</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/pdf/2406.04151" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/WooooDyy/AgentGym" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
   <tr align="center">
    <td>PEER: Expertizing Domain-Specific Tasks with a Multi-Agent Framework and Tuning Methods</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/pdf/2407.06985?" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/agentuniverse-ai/agentUniverse" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>BabyAGI</td>
    <td>2025</td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
    <td><a href="https://github.com/yoheinakajima/babyagi" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>OpenManus</td>
    <td>2025</td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
    <td><a href="https://github.com/mannaandpoem/OpenManus" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>AutoAgent: A Fully-Automated and Zero-Code Framework for LLM Agents</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2502.05957" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/HKUDS/AutoAgent" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>


## LAMs
<table><thead>
  <tr>
    <th>LAM Category</th>
    <th>Specific Models</th>
    <th>model</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td rowspan=10>Large Language Model</td>
    <td rowspan=5>GPT series</td>
    <td>GPT-1</td>
    <td>2020</td>
    <td><a href="https://hayate-lab.com/wp-content/uploads/2023/05/43372bfa750340059ad87ac8e538c53b.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/docs/transformers/model_doc/openai-gpt" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>GPT-2</td>
    <td>2023</td>
    <td><a href="https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/openai/gpt-2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>GPT-3</td>
    <td>2023</td>
    <td><a href="https://splab.sdu.edu.cn/GPT3.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>GPT-4</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2303.08774" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>OpenAI o1</td>
    <td>2024</td>
    <td><a href="https://cdn.openai.com/o1-system-card.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://www.github.com/openai/simple-evals" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Gemma series</td>
    <td>Gemma 1</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2403.08295" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>Gemma 2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2408.00118" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td rowspan=3>LLaMA series</td>
    <td>LLaMA-1</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2302.13971" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/meta-llama/llama-models" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LLaMA-2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2307.09288" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/collections/meta-llama/metas-llama2-models-675bfd70e574a62dd0e40541" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>LLaMA-3</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2407.21783" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>

 
  <tr align="center">
    <td rowspan=7>Large Vision Model</td>
    <td rowspan=2>SAM series</td>
    <td>SAM-1</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2304.02643" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/segment-anything" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>SAM-2</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2408.08315" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/sam2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>DINO series</td>
    <td>DINO V1</td>
    <td>2021</td>
    <td><a href="https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/dino" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>DINO V2</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2304.07193" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/dinov2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=3>Stable Diffusion series</td>
    <td> Stable Diffusion V1</td>
    <td>2022</td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/CompVis/latent-diffusion" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Stable Diffusion V2</td>
    <td>2022</td>
    <td><a href="https://doi.org/10.48550/arxiv.2204.11824" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Stability-AI/stablediffusion" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
   <td>Stable Diffusion V3</td>
   <td>2024</td>
   <td><a href="https://openreview.net/forum?id=FPnUhsQJ5B" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>


  
  <tr align="center">
    <td rowspan=4>Vision Language Model</td>
    <td>LLaVA</td>
    <td>LLaVA</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/haotian-liu/LLaVA" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td rowspan=2>Qwen-VL</td>
    <td> Qwen-VL</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2308.12966" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen-VL/tree/main" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Qwen-VL-Chat</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2308.12966" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://huggingface.co/Qwen/Qwen-VL-Chat/tree/main" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Mini-GPT4</td>
    <td>Mini-GPT4</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2304.10592" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Vision-CAIR/MiniGPT-4" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>


  
  <tr align="center">
    <td rowspan=4>Large Multimodal Model</td>
    <td rowspan=2>CoDi series</td>
    <td>CoDi-1</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/microsoft/i-Code/tree/main/i-Code-V3" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>CoDi-2</td>
    <td>2024</td>
    <td><a href="http://openaccess.thecvf.com/content/CVPR2024/html/Tang_CoDi-2_In-Context_Interleaved_and_Interactive_Any-to-Any_Generation_CVPR_2024_paper.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/microsoft/i-Code/tree/main/CoDi-2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Meta-Transformer</td>
    <td>Meta-Transformer</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2307.10802" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/invictus717/MetaTransformer" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>ImageBind</td>
    <td>ImageBind</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2305.05665" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/facebookresearch/ImageBind" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>

   
  <tr align="center">
    <td rowspan=3>World Model</td>
    <td>Sora</td>
    <td>Sora</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2402.17177" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
     <td>JEPA</td>
     <td>JEPA</td>
     <td>2022</td>
     <td><a href="https://openreview.net/pdf?id=BZ5a1r-kVsf" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td></td>
   </tr>
   <tr align="center">
     <td>Vista</td>
     <td>Vista</td>
     <td>2024</td>
     <td><a href="https://arxiv.org/abs/2405.17398" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://github.com/opendrivelab/vista" target="_blank" rel="noopener noreferrer">Code</a></td>
   </tr>


  <tr align="center">
    <td rowspan=3>Lightweight Large AI Model</td>
    <td>TinyLlama</td>
    <td>TinyLlama</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2401.02385" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jzhang38/TinyLlama" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
     <td>MobileVLM</td>
     <td>MobileVLM</td>
     <td>2024</td>
     <td><a href="https://arxiv.org/abs/2402.03766" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://github.com/Meituan-AutoML/MobileVLM" target="_blank" rel="noopener noreferrer">Code</a></td>
   </tr>
   <tr align="center">
      <td>Mini-Gemini</td>
      <td>Mini-Gemini</td>
      <td>2024</td>
      <td><a href="https://arxiv.org/abs/2403.18814" target="_blank" rel="noopener noreferrer">Paper</a></td>
      <td><a href="https://github.com/dvlab-research/MGM" target="_blank" rel="noopener noreferrer">Code</a></td>
    </tr>


  <tr align="center">
    <td rowspan=3>Large Reasoning Model</td>
   <tr align="center">
    <td>OpenAI o3-mini</td>
    <td>OpenAI o3-mini</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2501.17749" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
   <tr align="center">
     <td>DeepSeek</td>
     <td>DeepSeek-R1</td>
     <td>2025</td>
     <td><a href="https://arxiv.org/abs/2502.12893" target="_blank" rel="noopener noreferrer">Paper</a></td>
     <td><a href="https://github.com/dukeceicenter/jailbreak-o1o3-deepseek-r1" target="_blank" rel="noopener noreferrer">Code</a></td>
   </tr>
</tbody></table>


## Planner
<table><thead>
  <tr>
    <th>Title</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
 <tr align="center">
    <td>Generating SPARQL from Natural Language Using Chain-of-Thoughts Prompting</td>
    <td>2025</td>
    <td><a href="https://papers.dice-research.org/2024/SEMANTICS_Cot-SPARQL/public.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/dice-group/CoT-Sparql" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>Understanding Before Reasoning: Enhancing Chain-of-Thought with Iterative Summarization Pre-Prompting</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2501.04341?" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/zdhgreat/ISP-2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>Graph Chain-of-Thought: Augmenting Large Language Models by Reasoning on Graphs</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2404.07103" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/PeterGriffinJin/Graph-CoT" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2504.16563" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/cjj826/GoalAct" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Graph of thoughts: Solving elaborate problems with large language models</td>
    <td>2024</td>
    <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/29720" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/spcl/graph-of-thoughts" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Compositional Chain-of-Thought Prompting for Large Multimodal Models</td>
    <td>2025</td>
    <td><a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Mitra_Compositional_Chain-of-Thought_Prompting_for_Large_Multimodal_Models_CVPR_2024_paper.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/chancharikmitra/CCoT" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Thought Graph: Generating Thought Process for Biological Reasoning</td>
    <td>2024</td>
    <td><a href="https://dl.acm.org/doi/pdf/10.1145/3589335.3651572" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/ethan5437/thought-graph-www/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/cde328b7bf6358f5ebb91fe9c539745e-Paper-Conference.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/YangLing0818/buffer-of-thought-llm" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Step-by-Step Reasoning to Solve Grid Puzzles: Where do LLMs Falter?</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/pdf/2407.14790" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Mihir3009/GridPuzzle" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/00d80722b756de0166523a87805dd00f-Paper-Conference.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/sail-sg/CPO" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Tree of Thoughts: Deliberate Problem Solving with Large Language Models</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/pdf/2305.10601" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/kyegomez/tree-of-thoughts" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/pdf/2305.04091" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>ReAct: Synergizing Reasoning and Acting in Language Models</td>
    <td>2022</td>
    <td><a href="https://arxiv.org/abs/2210.03629" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/ysymyth/ReAct" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Least-to-Most Prompting Enables Complex Reasoning in Large Language Models</td>
    <td>2022</td>
    <td><a href="https://arxiv.org/pdf/2205.10625" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
</tbody></table>


## RAG/知识库
<table><thead>
  <tr>
    <th>Title</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>RAG and RAU: A Survey on Retrieval-Augmented Language Model in Natural Language Processing</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2404.19543" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/2471023025/RALM_Survey" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Emerging trends: a gentle introduction to RAG</td>
    <td>2024</td>
    <td><a href="https://www.cambridge.org/core/journals/natural-language-engineering/article/emerging-trends-a-gentle-introduction-to-rag/4FF461F4066A0C16135F2D2849E3356A" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/kwchurch/RAG" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>The Power of Noise: Redefining Retrieval for RAG Systems</td>
    <td>2024</td>
    <td><a href="https://dl.acm.org/doi/pdf/10.1145/3626772.3657834">Paper</a></td>
    <td><a href="https://github.com/florin-git/The-Power-of-Noise" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Retrieval-augmented generation for large language models: A survey</td>
    <td>2023</td>
    <td><a href="https://simg.baai.ac.cn/paperfile/25a43194-c74c-4cd3-b60f-0a1f27f8b8af.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Tongji-KGLLM/RAG-Survey" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/pdf/2401.15391" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/yixuantt/MultiHop-RAG" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2504.12330" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/ocean-luna/HMRAG" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Structured Review on RAG- and Multi-Agent Frameworks – Part II: Application-Based Assessment</td>
    <td>2025</td>
    <td><a href="https://www.essv.de/pdf/2025_43_50.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/BerriAI/litellm" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models</td>
    <td>2024</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/OSU-NLP-Group/HippoRAG" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2503.21322" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/LHRLAB/HyperGraphRAG" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>

## Memory
<table><thead>
  <tr>
    <th>Title</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>A Survey on the Memory Mechanism of Large Language Model based Agents</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/pdf/2404.13501" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/nuster1128/LLM_Agent_Memory_Survey" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>SURVEYFORGE: On the Outline Heuristics, Memory-Driven Generation,and Multi-dimensional Evaluation for Automated Survey Writing</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2503.04629?" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/Alpha-Innovator/SurveyForge" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Advances and challenges in foundation agents: From brain-inspired intelligence to evolutionary, collaborative, and safe systems</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2504.01990" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/FoundationAgents/awesome-foundation-agents" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>xlstm: Extended long short-term memory</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2405.04517" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/NX-AI/xlstm" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models</td>
    <td>2024</td>
    <td><a href="https://openreview.net/forum?id=hkujvAPVsg" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/OSU-NLP-Group/HippoRAG" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Persistent activity during working memory maintenance predicts long-term memory formation in the human hippocampus</td>
    <td>2024</td>
    <td><a href="https://www.cell.com/neuron/fulltext/S0896-6273(24)00661-5" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/rutishauserlab/SBCAT-NO-release-NWB" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>A-mem: Agentic memory for llm agents</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2502.12110" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/WujiangXu/AgenticMemory" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2504.19413" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://langchain-ai.github.io/langmem/" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Memory gym: Partially observable challenges to memory-based agents</td>
    <td>2023</td>
    <td><a href="https://openreview.net/forum?id=jHc8dCx6DDr" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/MarcoMeter/endless-memory-gym" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>

## Tools
<table><thead>
  <tr>
    <th>Title</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>CodeTF: One-stop Transformer Library for State-of-the-art Code LLM</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2306.00029" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/salesforce/CodeTF" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2401.14196" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/deepseek-ai/DeepSeek-Coder" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Lemur: Harmonizing Natural Language and Code for Language Agents</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2310.06830" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/OpenLemur/Lemur" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>HDDLGym: A Tool for Studying Multi-Agent Hierarchical Problems Defined in HDDL with OpenAI Gym</td>
    <td>2025</td>
    <td><a href="https://aair-lab.github.io/genplan25/papers/5.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/HDDLGym/HDDLGym" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2411.04905" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/richardodliu/OpenCodeEval" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2504.08066?" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/SakanaAI/AI-Scientist-v2" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
</tbody></table>


## Paper with code
<table><thead>
  <tr>
    <th>Title</th>
    <th>Release Time</th>
    <th>Link</th>
    <th>Download</th>
  </tr></thead>
<tbody>
  <tr align="center">
    <td>Openagents: An open platform for language agents in the wild</td>
    <td>2023</td>
    <td><a href="https://arxiv.org/abs/2310.10634" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
  <tr align="center">
    <td>Agentic ai: Autonomous intelligence for complex goals–a comprehensive survey</td>
    <td>2025</td>
    <td><a href="https://ieeexplore.ieee.org/document/10849561?denied=" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
  <tr align="center">
    <td>From llm reasoning to autonomous ai agents: A comprehensive review</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2504.19678" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
  <tr align="center">
    <td>Agentic ai for scientific discovery: A survey of progress, challenges</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2503.08979" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
  <tr align="center">
    <td>Agentic reasoning: Reasoning llms with tools for the deep research</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2502.04644" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/theworldofagents/Agentic-Reasoning" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Large language model enhanced multi-agent systems for 6g communications</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/pdf/2312.07850" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jiangfeibo/CommLLM" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Generative ai agents with large language model for satellite networks via a mixture of experts transmission</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/pdf/2404.09134" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/RickyZang/GAI-agent-satellite" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>A survey of agent interoperability protocols: Model context protocol (mcp), agent communication protocol (acp), agent-to-agent protocol (a2a), and agent network protocol (anp)</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2505.02279" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
  <tr align="center">
    <td>Reflexion: Language agents with verbal reinforcement learning</td>
    <td>2023</td>
    <td><a href="https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/noahshinn/reflexion" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Ai agents vs. agentic ai: A conceptual taxonomy, applications and challenge</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/pdf/2505.10468" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
   <tr align="center">
    <td>Cached model-as-a-resource: Provisioning large language model agents for edge intelligence in space-air-ground integrated networks</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2403.05826" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
   <tr align="center">
    <td>Self-resource allocation in multi-agent llm systems</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2504.02051" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
 <tr align="center">
    <td>Model context protocol-based internet of experts for wireless environment-aware llm agents</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2505.01834" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
 <tr align="center">
    <td>Enabling mobile ai agent in 6g era: Architecture and key technologies</td>
    <td>2024</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10599391" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
 <tr align="center">
    <td>Llm agents as 6g orchestrator: A paradigm for task-oriented physical-layer automation</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2410.03688" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
 <tr align="center">
    <td>When large language model agents meet 6g networks: Perception, grounding, and alignmen</td>
    <td>2024</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10648594" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
 <tr align="center">
    <td>Agent-driven generative semantic communication with cross-modality and prediction</td>
    <td>2025</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10815060" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="" target="_blank" rel="noopener noreferrer"></a></td>
  </tr>
 <tr align="center">
    <td>Wirelessagent: Large language model agents for intelligent wireless networks</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2505.01074" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/jwentong/WirelessAgent_R1" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Towards agentic ai networking in 6g: A generative foundation model-as-agent approach</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2503.15764" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>Llm-driven agentic ai approach to enhanced o-ran resilience in next-generation networks</td>
    <td>2025</td>
    <td><a href="https://www.techrxiv.org/doi/full/10.36227/techrxiv.174284755.59863143" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>Exploring llm-based multi-agent situation awareness for zero-trust space-air-ground integrated network</td>
    <td>2025</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10963886/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/OISF/suricat" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Airvista: Empowering uavs with 3d spatial reasoning abilities through a multimodal large language model agent</td>
    <td>2024</td>
    <td><a href="https://ieeexplore.ieee.org/abstract/document/10919532/" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
 <tr align="center">
    <td>Scenario-driven evaluation of autonomous agents: Integrating large language model for uav mission reliability</td>
    <td>2025</td>
    <td><a href="https://search.proquest.com/openview/45a54717a8c03da81300186f2960b30b/1?pq-origsite=gscholar&cbl=5046906" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>Task offloading with llm-enhanced multi-agent reinforcement learning in uav-assisted edge computing</td>
    <td>2025</td>
    <td><a href="https://search.proquest.com/openview/99adc0f117a249a1294b69c8b7e8172d/1?pq-origsite=gscholar&cbl=2032333" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
 <tr align="center">
    <td>Uav-codeagents: Scalable uav mission planning via multi-agent react and vision-language reasoning</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2505.07236" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
  <tr align="center">
    <td>Agentic retrievalaugmented generation: A survey on agentic rag</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2501.09136" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/asinghcsu/AgenticRAG-Survey" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>Agentnet: Decentralized evolutionary coordination for llm-based multi-agent systems</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2504.00587" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
 <tr align="center">
    <td>Usercentrix: An agentic memory-augmented ai framework for smart spaces</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2505.00472" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/evaluation/criteria/prompt.py" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>Multi-agent collaboration mechanisms: A survey of llms</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2501.06322" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/crewAIInc/crewAI" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
 <tr align="center">
    <td>Advancing multi-agent systems through model context protocol: Architecture, implementation, and applications</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2504.21030" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
   <tr align="center">
    <td>Building a secure agentic ai application leveraging a2a protocol</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2504.16902" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/kenhuangus/a2a-secure-coding-examples" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Agent-as-a-judge:Evaluate agents with agents</td>
    <td>2024</td>
    <td><a href="https://arxiv.org/abs/2410.10934" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td><a href="https://github.com/metauto-ai/agent-as-a-judge" target="_blank" rel="noopener noreferrer">Code</a></td>
  </tr>
  <tr align="center">
    <td>Survey on evaluation of llm-based agents</td>
    <td>2025</td>
    <td><a href="https://arxiv.org/abs/2503.16416" target="_blank" rel="noopener noreferrer">Paper</a></td>
    <td></td>
  </tr>
</tbody></table>

## The Team
Here is the list of our student contributors in each section.
| Section 	| Student Contributors 	|
|:-------	|:--------------------	|
|   The whole paper  | Zhengyu Du , Yuhan Zhang |
|   Literature Search   | Jian Zou , Dandan Qi  |
|   Project Maintenance   | Xitao Pan  |



## Contact Information for Source Code Submission or Update
If you intend to add or update the source code in the repository, please contact the following email addresses: jiangfb@hunnu.edu.cn, Dlj2017@hunnu.edu.cn, 240620854087@stu.hutb.edu.cn and 240620854065@stu.hutb.edu.cn.


## Update Log
| Version 	| Time 	| Update Content 	|
|:---	|:---	|:---	|
| v1 	| 2025/5/23 	| The initial version. 	|
| v2 	|  	|  Improve the writing.<br>Correct some minor errors.	 |
| v3 	|  	|  Improve the writing.<br>Correct some minor errors. 	|



## Citation   
```



```

