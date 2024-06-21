# SDXL LoRa Training
<a name="readme-top"></a>

# Table of contents
1. [Introduction](#introduction)
2. [Which Fine-Tuning Method?](#which-model)
3. [Used Technologies](#technologies)
4. [Used Hyperparameters](#hyperparameters)
4. [Getting Started - Colab Training](#colab)
4. [Getting Started - Streamlit](#streamlit)

<!-- ABOUT THE PROJECT -->
## Introduction <a name="introduction"></a>
In today's rapidly evolving AI landscape, the demand for high-quality, annotated datasets and customized models is greater than ever. To address this need, our project aims to develop an innovative module that seamlessly integrates data annotation with model fine-tuning. This module will leverage cutting-edge language models such as Claude or ChatGPT-4 to label provided customer images, and subsequently fine-tune a Stable Diffusion XL (SDXL) model using these annotations.

### Project Goal
The goal of this project is to create a robust and efficient module that automates the data annotation process and enhances model training. By integrating advanced AI capabilities, we aim to produce a highly adaptable and precise SDXL model fine-tuned with the newly annotated dataset.

### Key Objectives
1. Data Annotation:

* Utilize Claude/ChatGPT-4 (or similar) API to generate high-quality annotations for a set of 5-20 customer images.

* Ensure annotations are accurate and consistent to create a reliable dataset for training purposes.

2. Model Fine-Tuning:

* Fine-tune an SDXL model using the annotated dataset.

* Implement Low-Rank Adaptation (LoRA) techniques to optimize the model's performance based on the specific characteristics of the annotated data.

### Significance
This project is designed to enhance the efficiency and accuracy of AI model development by automating and integrating crucial processes. The combination of advanced language models for annotation and sophisticated fine-tuning methods will result in a highly effective solution that meets the growing needs of various AI applications. By reducing manual labor and improving model performance, this project will provide significant value to industries relying on generative AI technologies.

![sample_image](sample_images/sample_output.jpeg "Results")

## Which Fine-Tuning Method? <a name="which-model"></a>
### Prefered Method
There are diffrent types of fine-tuning. The choice of fine-tuning methods is depending on the system specifications and usage. LoRA (Low-Rank Adaptation) models offer greater efficiency and compactness. They function like adapters that build upon existing checkpoint models. Specifically, LoRA models update only a subset of parameters from the checkpoint model, thereby enhancing its capabilities. This approach allows LoRA models to maintain a smaller size, typically ranging from 2MB to 500MB, and enables frequent fine-tuning for specific concepts or styles.

For instance, when fine-tuning a Stable Diffusion model using DreamBooth, which modifies the entire model to adapt to a specific concept or style, significant computational resources are required due to the resulting large model size (approximately 2 to 7 GBs) and intensive GPU usage. In contrast, LoRA models achieve comparable inference results with significantly lower GPU requirements.

While LoRA is a widely adopted method, there are alternative approaches to modifying Stable Diffusion. One such method involves the crossattention module, which processes input derived from converting prompt text into text embeddings. Textual Inversions represent another approach, even more compact and faster than LoRA. However, Textual Inversions are limited to fine-tuning text embeddings alone for specific concepts or styles. The underlying U-Net responsible for image generation remains unchanged, restricting Textual Inversions to generating images similar to those used during training, without the ability to produce entirely new outputs.

In this project, there are two types of fine-tuning methods. First option is using the combination of DreamBooth and LoRa and the other is using only LoRa. Using the first option is the best choice and it is the prefered method in this project. The reasons for this choice are:
* Enhanced Adaptability: DreamBooth is a fine-tuning method that allows for comprehensive adaptation of the entire model to specific concepts or styles. By fine-tuning with DreamBooth, the SDXL model can learn nuanced details and characteristics that align closely with the desired outputs.

* Efficiency and Compactness: LoRA (Low-Rank Adaptation) comes into play after DreamBooth fine-tuning. LoRA models are designed to optimize efficiency by updating only a subset of the parameters of the checkpoint model. This approach significantly reduces the model size (typically 2MB to 500MB) compared to fully fine-tuned models, such as those modified solely by DreamBooth.

* Reduced Computational Resources: Combining DreamBooth with LoRA results in models that require fewer GPU resources during both training and inference. DreamBooth initially requires substantial resources due to its comprehensive fine-tuning process, but LoRA's subsequent parameter reduction ensures that the model remains manageable and efficient.

* Preservation of Performance: Despite its efficiency gains, LoRA maintains the high-quality performance achieved through DreamBooth fine-tuning. This combination ensures that the model retains its ability to generate impressive outputs, comparable to those produced by a fully fine-tuned model.

* Flexibility for Iterative Refinement: The iterative approach of DreamBooth followed by LoRA allows for iterative refinement and fine-tuning. This flexibility is crucial in scenarios where continuous adaptation to evolving concepts or styles is required without compromising the model's efficiency or performance.

### How it works?
As mentioned above, the prefered way of fine-tuning an SDXL model in this project is the combination DreamBooth and LoRa. The rationale behind combining DreamBooth and LoRA lies in optimizing the trade-off between model adaptability and computational efficiency. DreamBooth allows for thorough adaptation of the model's parameters to specific nuances in the data or desired outputs. However, this comprehensive adaptation can lead to larger model sizes and increased computational demands, especially during training and inference. On the other hand, LoRA intervenes post-DreamBooth to streamline the model, reducing its size while preserving its performance. This combination leverages the strengths of both approaches: DreamBooth for precise adaptation and LoRA for efficient parameter management.

The main steps of this fine-tuning approach are:
1. Parameter Adjustment: Use DreamBooth to adjust the entire set of parameters within the SDXL model to align more closely with the defined objectives. This process involves iterative updates based on the target dataset or desired output characteristics.

2. Training Phase: Execute the fine-tuning process using the defined objectives and training data. This phase ensures that the SDXL model becomes finely tuned to the specific nuances and requirements of the task at hand.

3. Parameter Selection: Post-DreamBooth, identify subsets of parameters that are most crucial for maintaining or enhancing performance. This step involves analyzing the importance and impact of different parameters within the fine-tuned SDXL model.

4. Low-Rank Factorization: Apply LoRA techniques, such as low-rank matrix factorization, to these selected parameter subsets. LoRA decomposes the parameter matrices into low-rank components, which reduces redundancy and focuses computational resources on the most influential parameters.

5. Selective Parameter Update: Update only the identified low-rank components, thereby optimizing the model's efficiency while preserving or improving its performance metrics.

## Used Technologies <a name="technologies"></a>
### Accelerate
Accelerate is a versatile and user-friendly library designed by Hugging Face to streamline and optimize the process of training and deploying machine learning models on a variety of hardware setups. It offers a unified interface that abstracts the complexities of configuring and managing different distributed training environments, such as multi-GPU and TPU setups. Accelerate makes it easy for developers to scale their PyTorch code, focusing on model development rather than the underlying infrastructure.

The benefits of this library are:
* Simplifies Distributed Training: Eliminates the need to manually configure and manage complex distributed setups, allowing you to focus on developing and fine-tuning your models.

* Enhances Resource Utilization: Maximizes the use of available hardware, ensuring that your GPUs and TPUs are utilized effectively to speed up training and inference processes.

* Supports Large Models: Facilitates the handling of large models that require significant memory and computational power, making it accessible to work with cutting-edge neural networks.

* Reduces Development Overhead: Streamlines the integration of distributed training into your projects, saving time and reducing the overhead associated with managing different hardware environments.

### Bitsandbytes
Bitsandbytes is an efficient and innovative library designed to optimize the performance of large-scale neural networks, particularly in the context of training and inference. It provides tools and techniques to significantly reduce memory consumption and computational overhead without sacrificing accuracy. One of the standout features of Bitsandbytes is its support for 8-bit precision optimizers.

In this project, we are using 8-bit optimizer technique. The 8-bit optimizer in Bitsandbytes is a technique that quantizes the precision of weights and gradients from 32-bit floating-point numbers to 8-bit integers during the training process. Utilizing the 8-bit optimizer from Bitsandbytes is especially advantageous for projects involving large-scale neural networks, such as the fine-tuning of generative models like Stable Diffusion XL (SDXL). The primary reasons to use the 8-bit optimizer include:

* Handling Larger Models: Fit larger models into available hardware memory, enabling the training of state-of-the-art architectures that would otherwise be infeasible.

* Faster Training: Accelerate training times by reducing the computational load, allowing for quicker experimentation and iteration.

* Resource Optimization: Maximize the use of available hardware, reducing the need for costly upgrades and making efficient use of existing resources.

* Enhanced Performance: Achieve comparable accuracy and performance to traditional 32-bit training methods while benefiting from the reduced memory and computational demands.

### Transformers
The Transformers library, developed by Hugging Face, is an open-source library that provides a wide range of state-of-the-art pre-trained models for natural language processing (NLP) and other tasks. It supports a variety of transformer architectures, such as BERT, GPT, T5, RoBERTa, and many others. The library is designed to make it easy to use these powerful models for a variety of applications, including text classification, translation, question answering, and more.

The benefits of using this library are:
* Access to Cutting-Edge Models: Easily access and implement some of the most advanced models in NLP and beyond, ensuring that you are working with top-performing architectures.

* Rapid Development: The library's straightforward API allows for quick prototyping and experimentation, significantly speeding up the development process.
Transfer Learning Capabilities: Fine-tune powerful pre-trained models on your specific tasks, leveraging prior knowledge and achieving high performance with less data.

* Versatile Applications: Use the library for a broad spectrum of tasks, including text classification, sentiment analysis, named entity recognition, machine translation, and more.

* Robust Community Support: Benefit from a strong community and extensive resources provided by Hugging Face, ensuring that you have the support needed to overcome challenges and innovate in your projects.

### PEFT
The PEFT (Parameter-Efficient Fine-Tuning) library is designed to optimize the fine-tuning process of large-scale machine learning models. Developed to address the challenges of fine-tuning massive models with limited computational resources, PEFT focuses on techniques that enable efficient adaptation of pre-trained models to new tasks with minimal changes to the model's parameters.

Benefits of using PEFT:
* Efficiency: Focuses on parameter-efficient methods, reducing the computational and memory overhead associated with fine-tuning large models.

* Accessibility: Makes it possible to fine-tune state-of-the-art models on standard hardware, democratizing access to advanced machine learning techniques.

* Speed: Accelerates the fine-tuning process, allowing for faster model updates and deployment.

* Performance: Ensures high performance even with fewer parameters being adjusted, thanks to advanced fine-tuning techniques.

* Versatility: Can be applied to a wide range of models and tasks, providing a versatile tool for machine learning practitioners.

## Used Hyperparameters <a name="hyperparameters"></a>
### Gradient Checkpointing
Backpropagation, which computes these gradients, requires storing intermediate activations of the model. This can be memory-intensive, especially for large models like SDXL. Gradient checkpointing addresses this memory challenge by trading off memory usage for additional computation time. Instead of storing all intermediate activations throughout the entire model during backpropagation, gradient checkpointing periodically recomputes activations starting from previously saved checkpoints. This approach reduces the peak memory usage by recomputing activations on-the-fly during the backward pass. By using gradient checkpointing, the memory overhead of storing all intermediate activations is reduced. This is particularly beneficial when fine-tuning models that have undergone extensive parameter adjustments (DreamBooth) and selective updates (LoRA). While gradient checkpointing reduces memory consumption, it introduces additional computational overhead due to recomputation. The trade-off between memory and computation needs to be balanced based on the available resources and the specific fine-tuning objectives.

### 8-Bit Adam
Adam (Adaptive Moment Estimation) is a popular optimization algorithm widely used in deep learning. It combines adaptive learning rates for each parameter with momentum to accelerate convergence. Normally, Adam uses 32-bit floating-point numbers (single precision) for storing gradients and updating parameters. '8 bit adam' modifies this by using 8-bit fixed-point numbers for these operations. By using 8-bit precision instead of 32-bit, '8 bit adam' significantly reduces the memory footprint required for storing gradients and parameters during training. This is particularly advantageous for large models like SDXL, which have numerous parameters. Lower precision arithmetic operations can potentially speed up computations due to reduced memory bandwidth requirements. This can lead to faster training times, especially on hardware architectures optimized for lower precision operations. However, there is trade-off. Using lower precision can affect the model's accuracy and stability, particularly if not implemented carefully. Techniques like gradient scaling or adaptive precision adjustment may be necessary to mitigate any potential accuracy loss.

### Mixed-Precision FP16
There are two precision levels in floating-point precision, 32-bit floating point and 16-bit floating point. 32-bit floating point is the standard precision used in most deep learning frameworks for storing model parameters and performing computations. It provides high numerical accuracy but requires more memory and computational resources. 16-bit floating point is the reduced precision format that uses half the memory of 32-bit floating point. It accelerates computations, especially on GPUs with tensor cores, while maintaining sufficient numerical precision for many deep learning tasks. Utilizing 16-bit precision can significantly speed up training times, especially for large models like SDXL that involve complex computations. Also, reduced precision requires less memory bandwidth, making it feasible to train larger models or batch sizes within available hardware limits.


<p align="right">(<a href="#readme-top">Back to Top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started - Colab Training <a name="colab"></a>
Instructions on setting up your project using Colab. Please follow the link below to try it out.

<a href="https://colab.research.google.com/github/nuwandda/sdxl-lora-training/blob/main/SDXL_DreamBooth_LoRA.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Getting Started - Streamlit Demo <a name="streamlit"></a>
To run the streamlit demo with the pretrained model trained with marble statue dataset, use the following commands.

### Install dependencies
To install the required packages, in a terminal, type:
  ```sh
  pip install -r requirements.txt
  ```

### Run demo
In a terminal, type:
  ```sh
  streamlit run main.py
  ```


<p align="right">(<a href="#readme-top">Back to Top</a>)</p>
