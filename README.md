# From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications
## Authors
### Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Octavia A. Dobre, Merouane Debbah
## Paper
### 
## Code
### https://github.com/jiangfeibo/ComAgent
## Abstract
With the advent of 6G communications, intelligent communication systems face multiple challenges, including constrained perception and response capabilities, limited scalability, and low adaptability in dynamic environments. This tutorial
provides a systematic introduction to the principles, design, and applications of Large Artificial Intelligence Models (LAMs) and Agentic AI technologies in intelligent communication systems, aiming to offer researchers a comprehensive overview of cuttingedge technologies and practical guidance. First, we outline the background of 6G communications, review the technological evolution from LAMs to Agentic AI, and clarify the tutorial’s motivation and main contributions. Subsequently, we present a comprehensive review of the key components required for constructing LAMs, including Transformers, Vision Transformers (ViTs), Variational AutoEncoders (VAEs), diffusion models, Diffusion Transformers (DiTs), and Mixture of Experts (MoEs). We further categorize LAMs and analyze their applicability, covering Large Language Models (LLMs), Large Vision Models (LVMs), Large Multimodal Models (LMMs), Large Reasoning Models (LRMs), and lightweight LAMs. Next, we propose a LAM-centric design paradigm tailored for communications, encompassing dataset construction and both internal and external learning approaches. Building upon this, we develop an LAM-based Agentic AI system for intelligent communications, clarifying its core components such as planners, knowledge bases, tools, and memory modules, as well as its interaction mechanisms, including both single-agent and multi-agent interactions. We also introduce a multi-agent framework with data retrieval, collaborative planning, and reflective evaluation for 6G. Subsequently, we provide a detailed overview of the applications of LAMs and Agentic AI in communication scenarios. Finally, we summarize the research challenges and future directions in current studies, aiming to support the development of efficient, secure, and sustainable next-generation intelligent communication systems.


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

