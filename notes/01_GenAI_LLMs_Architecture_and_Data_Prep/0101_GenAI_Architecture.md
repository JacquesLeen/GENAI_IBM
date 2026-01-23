# GEN AI Archtitecture

## Significance of Gen AI

GenAI refers to Deep Learning model that can generate content based on the data they were trained on. It comprehends different types of models

* **Text**
* **Image**
* **Audio**
* **Video**
* ...

an example is a GenAI model for text generation: it gets trained on a bunch of data, understands the pattern that link words together and when prompted provides a response based on the training it received. Similarly an Image model can be used to generate images from text, or from seed images. Lots of material can be created thanks to GenAI models, amongst others

* Content Creation
* Document Condensation
* Language Translation
* Chatbot and Assistants
* Data Analysis

## GenAI Architecture and Models

According to the task that you wish to implement there are different Architecture that can be implemented

* Recurrent Neural Networks (RNN): frequently used in NLP, language Translation, Spoeech Recognition, Image Captioning
* Transformers: Generative Pretrained Transformer (GPT) for generation of text
* General Adversarial Networks (GAN): Image and Video generation
* Variational Auto-Encoders (VAD): Art and Creative Design
* Diffusion Model: noise reduction

Different Archtiectures require different training approaches

|Model Type| Training Type|
|---|---|
|RNN| loop based|
|Transformers| self attention mechanism |
|GAN| competitive approach |
|VAE| characteristic based|
|Diffusion Model| statistical approach|

Reinforcement Learning techniques are in general heavily used to maximize rewards from a correct interaction with the environment for the Model.

## GenAI for NLP

GenAI can do a lot in terms of its utilization of Natural Language. It can:

* Comperehend Natural Language
* Generate human-like responses
* Understand Context (feelings, intentions)
* Provide meaningful conversation

The state of the art is due to the utilization of Deep Learning and then Transformers. Expecially those are really relevant in that they allow to impose a time-line like structure. GPT and ChatGPT are often used as examples of LLMs (Large Language Models) that are precisely twaeked for NLP. There are some differences between the two:

|Aspect| GPT| ChatGPT|
|---|---|---|
|Goal|Text Generation Tasks|Conversation Generation|
|Training|Supervised Learning| Combination of Supervised and Reinforcement Learning|
|Human Feedback|Not incorporated| Reinforcement Learning from Human Feedback (RLHF)|

The standard approach as to the training of LLMs nowadays is to leverage pretrained models and fine tune them for specific topics using smaller more oriented data sets

## Most Common Libraries for GenAI

* **PyTorch**: Open source DL framework, that allows dynamic computational graph, meaning that the network structure can change on the fly during execution

* **Tensorflow**: Open Source framework for ML and DL. It is designed with scalability in mind and this facilitates the training process and development of machine learning projects

* **Hugging Face**: HF is a platform that offers open source libraries with pre-trained models and tools to enable fine tuning and training GenAI models. It has an extensive catalog of models. For example the Transformers libraries contains pretrained models that work with text. Its Datasets library contains large scale data sets for training and evaluating models. Its tokenizers library helps with tokenization which is a key component of data preparation for GenAI

* **LangChain**: LC is an opensource framework that helps AI application development using LLMs by providing tools for prompt engineering and it is largely used in combination with Generative Pretrained Transformed (GPT)

* **Pydantic**: Pydantic is mainly used for data validation and type accuracy before processing which is crucial when working with large data sets
