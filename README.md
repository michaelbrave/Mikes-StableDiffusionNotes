- [Mikes-StableDiffusionNotes](#mikes-stablediffusionnotes)
  - [What is Stable Diffusion](#what-is-stable-diffusion)
    - [Origins and Research of Stable Diffusion](#origins-and-research-of-stable-diffusion)
      - [Initial Training Data](#initial-training-data)
      - [Core Technologies](#core-technologies)
      - [Tech That Stable Diffusion is Built On \& Technical Terms](#tech-that-stable-diffusion-is-built-on--technical-terms)
    - [Similar Technology / Top Competitors](#similar-technology--top-competitors)
      - [DALL-E2:](#dall-e2)
      - [Google's Imagen:](#googles-imagen)
      - [Midjourney:](#midjourney)
    - [Stable Diffusion Powered Websites and Communities](#stable-diffusion-powered-websites-and-communities)
      - [DreamStudio (Official by StabilityAI):](#dreamstudio-official-by-stabilityai)
      - [PlaygroundAI:](#playgroundai)
      - [LeonardoAI:](#leonardoai)
      - [NightCafe:](#nightcafe)
      - [BlueWillow:](#bluewillow)
      - [DreamUp By DeviantArt:](#dreamup-by-deviantart)
      - [Lexica:](#lexica)
      - [Dreamlike Art:](#dreamlike-art)
      - [Art Breeder Collage Tool:](#art-breeder-collage-tool)
      - [Dream by Wombo:](#dream-by-wombo)
    - [Community Chatrooms and Gathering Locations](#community-chatrooms-and-gathering-locations)
  - [Basics, Settings and Operations](#basics-settings-and-operations)
  - [What Can Be Done With Stable Diffusion](#what-can-be-done-with-stable-diffusion)
    - [Core Functionality \& Use Cases](#core-functionality--use-cases)
      - [Character Design](#character-design)
      - [Video Game Asset Creation](#video-game-asset-creation)
      - [Architecture and Interior Design](#architecture-and-interior-design)
    - [Use Cases Other Than Image Generation](#use-cases-other-than-image-generation)
      - [Video \& Animation](#video--animation)
        - [Deforum](#deforum)
        - [Depth Module for Stable Diffusion](#depth-module-for-stable-diffusion)
        - [Gen1](#gen1)
      - [3D Generation Techniques for Stable Diffusion \& Related Diffusion Based 3D Generation](#3d-generation-techniques-for-stable-diffusion--related-diffusion-based-3d-generation)
      - [Text to 3D](#text-to-3d)
      - [DMT Meshes / Point Cloud Based](#dmt-meshes--point-cloud-based)
      - [3D radiance Fields](#3d-radiance-fields)
      - [Novel View Synthesis](#novel-view-synthesis)
        - [NeRF Based:](#nerf-based)
        - [Img to Fspy to Blender:](#img-to-fspy-to-blender)
        - [Image to Shapes](#image-to-shapes)
      - [3D Texturing Techniques for Stable Diffusion](#3d-texturing-techniques-for-stable-diffusion)
        - [Using Stable Diffusion for 3D Texturing:](#using-stable-diffusion-for-3d-texturing)
        - [Dream Textures:](#dream-textures)
      - [Music](#music)
        - [Riffusion](#riffusion)
      - [Image-Based Mind Reading](#image-based-mind-reading)
      - [Synthetic Data Creation](#synthetic-data-creation)
  - [How it Works](#how-it-works)
  - [Beginner's How To](#beginners-how-to)
  - [Popular UI's](#popular-uis)
    - [Automatic 1111](#automatic-1111)
      - [Automatic 1111 Extensions](#automatic-1111-extensions)
    - [Kohya](#kohya)
      - [Addons](#addons)
    - [EasyDiffusion / Formerly Stable Diffusion UI](#easydiffusion--formerly-stable-diffusion-ui)
    - [InvokeAI](#invokeai)
    - [DiffusionBee (Mac OS)](#diffusionbee-mac-os)
    - [NKMD GUI](#nkmd-gui)
    - [ComfyUi](#comfyui)
    - [AINodes](#ainodes)
  - [Model and Training UI's](#model-and-training-uis)
    - [Other Sofware Addons that Act like a UI](#other-sofware-addons-that-act-like-a-ui)
  - [Hardware Requirements and Cloud-Based Solutions](#hardware-requirements-and-cloud-based-solutions)
    - [Xformers for Stable Diffusion](#xformers-for-stable-diffusion)
  - [Resources](#resources)
    - [Communities/Sites to Use it](#communitiessites-to-use-it)
    - [Where to Get Models](#where-to-get-models)
    - [Communities/Sites to discuss and share things with it](#communitiessites-to-discuss-and-share-things-with-it)
    - [Sites for Prompt Inspiration for Stable Diffusion](#sites-for-prompt-inspiration-for-stable-diffusion)
  - [Stable Diffusion (SD) Core and Models](#stable-diffusion-sd-core-and-models)
    - [Base Models for Stable Diffusion](#base-models-for-stable-diffusion)
      - [Stable Diffusion Models 1.4 and 1.5](#stable-diffusion-models-14-and-15)
      - [Stable Diffusion Models 2.0 and 2.1](#stable-diffusion-models-20-and-21)
        - [512-Depth Model for Image-to-Image Translation](#512-depth-model-for-image-to-image-translation)
    - [VAE (Variational Autoencoder) in Stable Diffusion](#vae-variational-autoencoder-in-stable-diffusion)
      - [Original Autoencoder in Stable Diffusion](#original-autoencoder-in-stable-diffusion)
      - [EMA VAE in Stable Diffusion](#ema-vae-in-stable-diffusion)
      - [MSE VAE in Stable Diffusion](#mse-vae-in-stable-diffusion)
    - [Samplers](#samplers)
      - [Ancestral Samplers](#ancestral-samplers)
        - [DPM++ 2S A Karras](#dpm-2s-a-karras)
        - [DPM++ A](#dpm-a)
        - [Euler A](#euler-a)
        - [DPM Fast](#dpm-fast)
        - [DPM Adaptive](#dpm-adaptive)
      - [DPM++](#dpm)
        - [DPM++ SDE](#dpm-sde)
        - [DPM++ 2M](#dpm-2m)
    - [Community Models](#community-models)
      - [Fine Tuned](#fine-tuned)
      - [Merged/Merges](#mergedmerges)
        - [Tutorial for Add Difference Method](#tutorial-for-add-difference-method)
      - [Megamerged/MegaMerges](#megamergedmegamerges)
      - [Embeddings](#embeddings)
      - [Community Forks](#community-forks)
  - [Capturing Concepts / Training](#capturing-concepts--training)
    - [Image2Text](#image2text)
      - [CLIP Interrogation](#clip-interrogation)
      - [BLIP Captioning](#blip-captioning)
      - [DanBooru Tags / Deepdanbooru](#danbooru-tags--deepdanbooru)
      - [Waifu Diffusion 1.4 tagger - Using DeepDanBooru Tags](#waifu-diffusion-14-tagger---using-deepdanbooru-tags)
    - [Dataset and Image Preparation](#dataset-and-image-preparation)
      - [Captioning](#captioning)
      - [Regularization/Classifier Images](#regularizationclassifier-images)
    - [Training](#training)
      - [File Type Overview](#file-type-overview)
      - [Textual Inversion](#textual-inversion)
        - [Negative Embedding](#negative-embedding)
      - [Hypernetworks](#hypernetworks)
      - [LORA](#lora)
        - [LoHa](#loha)
      - [Aescetic Gradients](#aescetic-gradients)
    - [Fine Tuning / Checkpoints/Diffusers/Safetensors](#fine-tuning--checkpointsdiffuserssafetensors)
      - [Token Based](#token-based)
        - [Dreambooth](#dreambooth)
        - [Custom Diffusion by Adobe](#custom-diffusion-by-adobe)
      - [Caption Based Fine Tuning](#caption-based-fine-tuning)
        - [Fine Tuning](#fine-tuning)
      - [Decoding Checkpoints](#decoding-checkpoints)
    - [Mixing](#mixing)
      - [Using Multiple types of models and embeddings](#using-multiple-types-of-models-and-embeddings)
        - [Multiple Embeddings](#multiple-embeddings)
        - [Multiple Hypernetworks](#multiple-hypernetworks)
        - [Multiple LORA's](#multiple-loras)
      - [Merging](#merging)
        - [Merging Checkpoints](#merging-checkpoints)
      - [Converting](#converting)
    - [One Shot Learning \& Similar](#one-shot-learning--similar)
      - [DreamArtist (WebUI Extension)](#dreamartist-webui-extension)
      - [Universal Guided Diffusion](#universal-guided-diffusion)
  - [Initiating Composition](#initiating-composition)
    - [Text2Image](#text2image)
      - [Notes on Resolution](#notes-on-resolution)
      - [Prompt Editing](#prompt-editing)
      - [Negative Prompts](#negative-prompts)
      - [Alternating Words](#alternating-words)
      - [Prompt Delay](#prompt-delay)
      - [Prompt Weighting](#prompt-weighting)
      - [Ui specific Syntax](#ui-specific-syntax)
    - [Exploring](#exploring)
      - [Randomness](#randomness)
        - [Random Words](#random-words)
        - [Wildcards](#wildcards)
      - [Brute Force](#brute-force)
        - [Prompt Matrix](#prompt-matrix)
        - [XY Grid](#xy-grid)
        - [One Parameter](#one-parameter)
  - [Editing Composition](#editing-composition)
    - [Image2Image](#image2image)
      - [Img2Img](#img2img)
      - [Inpainting](#inpainting)
      - [Outpainting](#outpainting)
      - [Loopback](#loopback)
      - [InstructPix2Pix](#instructpix2pix)
      - [Depth2Image](#depth2image)
        - [Depth Map](#depth-map)
        - [Depth Preserving Img2Img](#depth-preserving-img2img)
      - [ControlNet](#controlnet)
    - [Pix2Pix-zero](#pix2pix-zero)
    - [Seed Resize](#seed-resize)
      - [Variations](#variations)
  - [Finishing](#finishing)
    - [Upscaling](#upscaling)
      - [BSRGAN](#bsrgan)
      - [ESRGAN](#esrgan)
        - [4x RealESRGAN](#4x-realesrgan)
        - [Lollypop](#lollypop)
        - [Universal Upscaler](#universal-upscaler)
        - [Ultrasharp](#ultrasharp)
        - [Uniscale](#uniscale)
        - [NMKD Superscale](#nmkd-superscale)
        - [Remacri by Foolhardy](#remacri-by-foolhardy)
      - [SD Upscale](#sd-upscale)
        - [SD 2.0 4xUpscaler](#sd-20-4xupscaler)
    - [Restoring](#restoring)
      - [Face Restoration](#face-restoration)
        - [GFPGAN](#gfpgan)
        - [Code Former](#code-former)
  - [Software Addons](#software-addons)
    - [Blender Addons](#blender-addons)
      - [Blender ControlNet](#blender-controlnet)
      - [Makes Textures / Vision](#makes-textures--vision)
      - [OpenPose](#openpose)
      - [OpenPose Editor](#openpose-editor)
      - [Dream Textures](#dream-textures-1)
      - [AI Render](#ai-render)
      - [Stability AI's official Blender](#stability-ais-official-blender)
      - [CEB Stable Diffusion (Paid)](#ceb-stable-diffusion-paid)
      - [Cozy Auto Texture](#cozy-auto-texture)
    - [Blender Rigs/Bones](#blender-rigsbones)
      - [ImpactFrames' OpenPose Rig](#impactframes-openpose-rig)
      - [ToyXYZ's Character bones that look like Openpose for blender](#toyxyzs-character-bones-that-look-like-openpose-for-blender)
      - [3D posable Mannequin Doll](#3d-posable-mannequin-doll)
      - [Riggify model](#riggify-model)
    - [Maya](#maya)
      - [ControlNet Maya Rig](#controlnet-maya-rig)
    - [Photoshop](#photoshop)
      - [Stable.Art](#stableart)
      - [Auto Photoshop Plugin](#auto-photoshop-plugin)
    - [Daz](#daz)
      - [Daz Control Rig](#daz-control-rig)
    - [Cinema4D](#cinema4d)
      - [Colors Scene (possibly no longer needed since controlNet Update)](#colors-scene-possibly-no-longer-needed-since-controlnet-update)
  - [Related Technologies, Communities and Tools, not necessarily Stable Diffusion, but Adjacent](#related-technologies-communities-and-tools-not-necessarily-stable-diffusion-but-adjacent)
  - [Techniques \& Possibilities](#techniques--possibilities)
    - [Clip Skip \& Alternating](#clip-skip--alternating)
    - [Multi Control Net and blender for perfect Hands](#multi-control-net-and-blender-for-perfect-hands)
    - [Blender to Depth Map](#blender-to-depth-map)
      - [Blender to depth map for concept art](#blender-to-depth-map-for-concept-art)
      - [depth map for terrain and map generation?](#depth-map-for-terrain-and-map-generation)
    - [Blender as Camera Rig](#blender-as-camera-rig)
    - [SD depthmap to blender for stretched single viewpoint depth perception model](#sd-depthmap-to-blender-for-stretched-single-viewpoint-depth-perception-model)
    - [Daz3D for posing](#daz3d-for-posing)
    - [Mixamo for Posing](#mixamo-for-posing)
    - [Figure Drawing Poses as Reference Poses](#figure-drawing-poses-as-reference-poses)
    - [Generating Images to turn into 3D sculpting brushes](#generating-images-to-turn-into-3d-sculpting-brushes)
    - [Stable Diffusion to Blender to create particles using automesh plugin](#stable-diffusion-to-blender-to-create-particles-using-automesh-plugin)
  - [Not Stable Diffusion But Relevant Techniques](#not-stable-diffusion-but-relevant-techniques)
  - [Other Resources](#other-resources)
    - [API's](#apis)

# Mikes-StableDiffusionNotes
Notes on Stable Diffusion: An attempt at a comprehensive list

The following is a list of stable diffusion tools and resources compiled from personal research and understanding, with a focus on what is possible to do with this technology while also cataloging resources and useful links along with explanations. Please note that an item or link listed here is not a recommendation unless stated otherwise. Feedback, suggestions and corrections are welcomed and can be submitted through a pull request or by contacting me on Reddit (https://www.reddit.com/user/mikebrave) or Discord (MikeBrave#6085).




## What is Stable Diffusion

Stable Diffusion is an open-source machine learning model that can generate images from text, modify images based on text or enhance low-resolution or low-detail images. It has been trained on billions of images and can produce results that are on par with those generated by DALL-E 2 and MidJourney.

Stable Diffusion (SD) is a deep-learning, text-to-image model that was released in 2022. Its primary function is to generate detailed images based on text descriptions. The model uses a combination of random static generation, noise, and pattern recognition through neural nets that are trained on keyword pairs. These pairs correspond to patterns found in a given training image that match a particular keyword.

To generate an image, the user inputs a text description, and the SD model references the keyword pairs associated with the words in the description. The model then produces a shape that corresponds to the patterns identified in the image. Over several passes, the image becomes clearer and eventually results in a final image that matches the text prompt.

Stable Diffusion is a latent diffusion model, which is a type of deep generative neural network. It was developed by the CompVis group at LMU Munich in collaboration with Stability AI, Runway, EleutherAI, and LAION. In October 2022, Stability AI raised US$101 million in a round led by Lightspeed Venture Partners and Coatue Management.

Stable Diffusion's code and model weights have been released publicly, and it can run on most consumer hardware equipped with a modest GPU with at least 8 GB VRAM. This marks a departure from previous proprietary text-to-image models such as DALL-E and Midjourney, which were accessible only via cloud services.

To better understand Stable Diffusion and how it works, there are several visual guides available. Jalammar's blog (https://jalammar.github.io/illustrated-stable-diffusion/) provides an illustrated guide to the model, while the Stable Diffusion Art website (https://stable-diffusion-art.com/how-stable-diffusion-work/) offers a step-by-step breakdown of the process.

In addition, a Colab notebook (https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing) is available to allow users to experiment with and gain a deeper understanding of the Stable Diffusion model.
  
Wikiepedia: https://en.wikipedia.org/wiki/Stable_Diffusion  
source code: https://github.com/justinpinkney/stable-diffusion  
Homepage: https://stability.ai/  



### Origins and Research of Stable Diffusion

Stable Diffusion (SD) is a deep-learning, text-to-image model that was released in 2022. It was developed by the CompVis group at LMU Munich in collaboration with Stability AI, Runway, EleutherAI, and LAION. The model was created through extensive research into deep generative neural networks and the diffusion process.

In the original announcement (https://stability.ai/blog/stable-diffusion-announcement), the creators of SD outlined the model's key features and capabilities. These include the ability to generate high-quality images based on text descriptions, as well as the flexibility to be applied to other tasks such as inpainting and image-to-image translation.

Stable Diffusion is a latent diffusion model, which is a type of deep generative neural network that uses a process of random noise generation and diffusion to create images. The model is trained on large datasets of images and text descriptions to learn the relationships between the two. This training process involves extensive experimentation and optimization to ensure that the model can accurately generate images based on text prompts.

The source code for Stable Diffusion is publicly available on GitHub (https://github.com/CompVis/stable-diffusion). This allows researchers and developers to experiment with the model, contribute to its development, and use it for their own projects.

Stability AI, the primary sponsor of Stable Diffusion, raised US$101 million in October 2022 to support further research and development of the model. The success of the model has highlighted the potential of deep learning and generative neural networks in the field of computer vision and image generation.

https://research.runwayml.com/the-research-origins-of-stable-difussion

#### Initial Training Data
LAION-5B - 5 billion image-text pairs were classified based on language and filtered into separate datasets by resolution  
Laion-Aesthetics v2 5+  

#### Core Technologies
Variational Autoencoder (VAE)  
- The simplest explanation is that it makes an image small then makes it bigger again. 
- A Variational Autoencoder (VAE) is an artificial neural network architecture that belongs to the families of probabilistic graphical models and variational Bayesian methods. It is a type of neural network that learns to reproduce its input, and also map data to latent space. VAEs use probability modeling in a neural network system to provide the kinds of equilibrium that autoencoders are typically used to produce. The neural network components are typically referred to as the encoder and decoder for the first and second component respectively. VAE's are part of the neural network model that encodes and decodes the images to and from the smaller latent space, so that computation can be faster. Any models you use, be it v1, v2 or custom, already comes with a default VAE


U-Net  
-  U-Net is used in Stable Diffusion to reduce the noise (denoises) in the image using the text prompt as a conditional. The U-Net model is used in the diffusion process to generate images.  The network is based on the fully convolutional network and its architecture was modified and extended to work with fewer training images and to yield more precise segmentations.
- In the case of image segmentation, the goal is to classify each pixel of an image into a specific class. For example, in medical imaging, the goal is to classify each pixel of an image into a specific organ or tissue type. U-Net is used to perform image segmentation by taking an image as input and outputting a segmentation map that classifies each pixel of the input image into a specific class
- U-Net is designed to work with fewer training images by using data augmentation to use the available annotated samples more efficiently
-  The architecture of U-Net is also designed to yield more precise segmentations by using a contracting path to capture context and a symmetric expanding path that enables precise localization


Text Encoder  
- Stable Diffusion is a latent diffusion model conditioned on the (non-pooled) text embeddings of a CLIP ViT-L/14 text encoder1. The text encoder is used to turn your prompt into a latent vector
- In the context of machine learning, a latent vector is a vector that represents a learned feature or representation of a data point that is not directly observable. For example, in the case of Stable Diffusion, the text encoder is used to turn your prompt into a latent vector that represents a learned feature or representation of the prompt that is not directly observable.

#### Tech That Stable Diffusion is Built On & Technical Terms  
Transformers  
- A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision (CV)
- Transformers are neural networks that learn context and understanding through sequential data analysis. The Transformer models use a modern and evolving mathematical techniques set, generally known as attention or self-attention. This set helps identify how distant data elements influence and depend on one another


LLM  
- LLM stands for Large Language Model. Large language models are a type of neural network that can generate human-like text by predicting the probability of the next word in a sequence of words. a good example of this would be ChatGPT


VQGAN  
- VQGAN is short for Vector Quantized Generative Adversarial Network and is utilized for high-resolution images; and is a type of neural network architecture that combines convolutional neural networks with Transformers. VQGAN employs the same two-stage structure by learning an intermediary representation before feeding it to a transformer. However, instead of downsampling the image, VQGAN uses a codebook to represent visual parts.
- https://compvis.github.io/taming-transformers/


Diffusion Models  
- a simple explanation is that it uses noising and denoising to learn how to reconstruct images.
- Diffusion models are a class of generative models used in machine learning to learn the latent structure of a dataset by modeling the way in which data points diffuse through the latent space1. They are Markov chains trained using variational inference1. The goal of diffusion models is to generate data similar to the data on which they are trained by destroying training data through the successive addition of Gaussian noise, and then learning to recover the data by reversing this noising process2.
- Diffusion models have emerged as a powerful new family of deep generative models with record-breaking performance in many applications, including image synthesis, video generation, and molecule design


Latent Diffusion Models  
- Latent diffusion models are machine learning models designed to learn the underlying structure of a dataset by mapping it to a lower-dimensional latent space. This latent space represents the data in which the relationships between different data points are more easily understood and analyzed1. Latent diffusion models use an auto-encoder to map between image space and latent space. The diffusion model works on the latent space, which makes it a lot easier to train2. Our latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs


CLIP  
- CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similar to the zero-shot capabilities of GPT-2 and 31. CLIP is much more efficient and achieves the same accuracy roughly 10x faster2. Because they learn a wide range of visual concepts directly from natural language, CLIP models are significantly more flexible and general than existing ImageNet models
- https://research.runwayml.com/


Gaussian Noise
- the simplest way to explain it is random static that get's used a lot for things we want randomness for.
- Gaussian noise is a term from signal processing theory denoting a kind of signal noise that has a probability density function (pdf) equal to that of the normal distribution (which is also known as the Gaussian distribution)1. Gaussian noise is a statistical noise having a probability density function equal to normal distribution, also known as Gaussian Distribution. Random Gaussian function is added to Image function to generate this noise2. Gaussian noise is a type of noise that follows a Gaussian distribution. A Gaussian filter is a tool for de-noising, smoothing and blurring


Denoising Autoencoders  
- A Denoising Autoencoder (DAE) is a type of autoencoder, which is a type of neural network used for unsupervised learning. The DAE is used to remove noise from data, making it better for analysis. The DAE works by taking a noisy input signal and encoding it into a smaller representation, removing the noise. The smaller representation is then decoded back into the original input signal1. Denoising autoencoders are a stochastic version of standard autoencoders that reduces the risk of learning the identity function2. Specifically, if the autoencoder is too big, then it can just learn the data, so the output equals the input, and does not perform any useful representation learning or dimensionality reduction


ResNet  
- ResNet, short for Residual Network is a specific type of neural network that was introduced in 2015 by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun in their paper “Deep Residual Learning for Image Recognition”. The ResNet models were extremely successful which you can guess from the following: ResNet won the ImageNet and COCO 2015 competitions, and its variants were the foundations of the first places in all five main tracks of the ImageNet and COCO 2016 competitions1. A Residual Neural Network (ResNet) is an Artificial Neural Network (ANN) of a kind that stacks residual blocks on top of each other to form a network2. ResNet is a deep neural network that is capable of learning thousands of layers


Latent Space  
- Latent space, also known as a latent feature space or embedding space, is an embedding of a set of items within a manifold in which items resembling each other are positioned closer to one another in the latent space. Position within the latent space can be viewed as being defined by a set of latent variables that emerge from the resemblances between the items1. If I have to describe latent space in one sentence, it simply means a representation of compressed data2. Latent space is a concept in machine learning and deep learning that refers to the space of latent variables that are learned by a model


Watermark Detection  
- The creators of LAION-5B trained a watermark detection model and used it to calculate confidence scores for every image in LAION-5B
- https://github.com/LAION-AI/LAION-5B-WatermarkDetection  




### Similar Technology / Top Competitors

Stable Diffusion (SD) is a cutting-edge text-to-image generation model that has been receiving significant attention since its release in 2022. However, there are other similar technologies and programs that have also been developed for this purpose. Some of the most notable ones are:

#### DALL-E2:
This is a text-to-image model developed by OpenAI that is similar to Stable Diffusion in its approach. It uses a transformer architecture and a discrete VAE to generate high-quality images based on text prompts.  
https://openai.com/product/dall-e-2  

#### Google's Imagen:
This is a machine learning system developed by Google that generates realistic images from textual descriptions. It uses a combination of neural networks and computer vision algorithms to create images that match the text prompt. This has yet to be released to the public.  
https://imagen.research.google/  

#### Midjourney:
This is a text-to-image model developed by a team of researchers at Peking University that generates images from textual descriptions. It uses a combination of attention mechanisms and adversarial training to generate high-quality images that match the text input.  
https://www.midjourney.com/  

Each of these programs uses different approaches and techniques to generate images from text descriptions. While Stable Diffusion has received significant attention recently, these other programs offer alternative methods for generating images based on text prompts.  


### Stable Diffusion Powered Websites and Communities
Some of the most notable websites and communities based on SD are:

#### DreamStudio (Official by StabilityAI):
This website uses Stable Diffusion to generate high-quality images based on user-submitted text prompts. It offers a simple and intuitive user interface and allows users to download or share their generated images.  
https://dreamstudio.ai/  

#### PlaygroundAI:
This is an online community that focuses on exploring the capabilities of Stable Diffusion and other deep-learning models. It provides a platform for researchers and enthusiasts to share their work, collaborate on projects, and discuss the latest developments in the field.  
https://playgroundai.com/  

#### LeonardoAI:
This is an online community that uses Stable Diffusion and other AI models to generate high-quality art and design. It provides a platform for artists and designers to experiment with new tools and techniques and showcase their work to a wider audience.  
https://app.leonardo.ai/  

#### NightCafe:
This website uses Stable Diffusion to generate surreal and dreamlike images based on user-submitted text prompts. It offers a unique and creative approach to image generation and has gained a dedicated following among art enthusiasts.  
https://nightcafe.studio/  

#### BlueWillow:
This is a design studio that uses Stable Diffusion and other deep-learning models to generate unique and creative designs for clients. It offers a range of services, including branding, website design, and digital art, and has gained a reputation for its innovative use of AI in design.  
https://www.bluewillow.ai/  

#### DreamUp By DeviantArt:  
DreamUp is an image-generation tool powered by your prompts that allows you to visualize most anything you can DreamUp! It is operated by DeviantArt, Inc. and is designed to create AI art knowing that creators and their work are treated fairly. You can create any image you can imagine with the power of artificial intelligence! You can try DreamUp with 5 free prompts.g. DeviantArt CEO Moti Levy says that the site isn’t doing any DeveintArt-specific training for DreamUp and that the tool is Stable Diffusion.  
https://www.deviantart.com/dreamup  

#### Lexica:  
Lexica is a self-styled stable diffusion search engine, it is a web app that provides access to a massive database of AI-generated images and their accompanying text prompts. It features a simple search box and discord link, a grid layout mode to view hundreds of images on one page, and a slider to change the size of the image previews. It also has image generation capabilities which can be especially useful when finding a prompt you like that you would like to immediately try.  
https://lexica.art/  

#### Dreamlike Art:  
Dreamlike.art that lets you generate free AI art straight from their website. It features a “Infinity Canvas” feature which allows you to outpaint images. This lets you create images larger than usual and can result in some amazing panoramic-style pictures  
https://dreamlike.art/  
https://www.reddit.com/r/DreamlikeArt/  

#### Art Breeder Collage Tool:  
Artbreeder Collage is a structured image generation tool with prompts and simple drawing tools. It allows mixing different pictures and shapes you can choose from the library or draw yourself with a text prompt to generate new art with the power of neural networks12. You can start with a collage that someone else has already created and make your own tweaks by moving, resizing and changing the colors of elements or by adding new ones. Or you can start out from scratch, either using a text prompt generated by the platform or by writing your own  
https://www.artbreeder.com/browse  

#### Dream by Wombo:
This is a mobile application that uses Stable Diffusion to generate animated images based on user-submitted audio prompts. It has gained significant popularity for its ability to create humorous and entertaining animations.  
https://dream.ai/  


This is not a comprehensive list, there are many other websites and communities that use Stable Diffusion and other text-to-image models. Please contribute to this list.  

### Community Chatrooms and Gathering Locations
Reddit Core Communities
- /r/StableDiffusion
- /r/sdforall
- /r/dreambooth
- /r/stablediffusionUI
- /r/civitai  

Reddit Related Communities
- /r/aiArt
- /r/AIArtistWorkflows
- /r/aigamedev
- /r/AItoolsCatalog
- /r/artificial
- /r/bigsleep
- /r/deepdream
- /r/dndai
- /r/dreamlikeart
- /r/MediaSynthesis

Discord
- Stable Foundation https://discord.gg/stablediffusion






## Basics, Settings and Operations

different sample methods  

sample steps  

CFG Scale
- https://arxiv.org/abs/2112.10741

denoising settings  


## What Can Be Done With Stable Diffusion

### Core Functionality & Use Cases
Stable diffusion is primarily used for image generation, upscaling images and editing images. Subsets of these activities could be style transfer, photo repair, color or texture filling, image completion or polishing, and image variation.  

- Image Generation
- Upscaling Images
- Editing Images
- Style Transfer
- Photo Repair/Touchups
- Color/Texture Filling
- Image Completion/Polishing
- Image Variation
- Outpainting

#### Character Design

#### Video Game Asset Creation

#### Architecture and Interior Design

### Use Cases Other Than Image Generation

#### Video & Animation

##### Deforum

##### Depth Module for Stable Diffusion

Stable Diffusion (SD) is a powerful text-to-image generation model that can be used for a wide range of applications. To generate videos with a 3D perspective, a Depth Module has been developed that adds a mesh generation capability to SD.

The Depth Module can be accessed through the Github repository (https://github.com/thygate/stable-diffusion-webui-depthmap-script). To generate the mesh required for video generation, the user needs to enable the "Generate 3D inpainted mesh" option on the Depth tab. This option can take several minutes to an hour, depending on the size of the image being processed. Once completed, the mesh in PLY format and four demo videos are generated, and all files are saved to the extras directory.

The Depth Module also allows for the generation of videos from the PLY mesh on the Depth tab. This option requires the mesh created by the extension, as files created elsewhere might not work correctly. Some additional information is stored in the file that is required for the video generation process, such as the required value for dolly. Most options are self-explanatory and can be adjusted to achieve the desired results.

The Depth Module is a useful extension to Stable Diffusion that enables users to create videos with a 3D perspective. It requires some additional processing time, but the results can be impressive and add a new dimension to the images generated by the model.

##### Gen1
though not publicly released and technically separate from stable diffusion, it is created by the same company and original authors of stable diffusion and we can assume that a lot of the technology under the hood is similar if not the same. But a note about it should be included here.

Gen1 takes a video and a style image and applies that style to that image, this allows for things like a video of stacks of boxes to be turned into a cityscape or things like that.   
https://research.runwayml.com/gen1 

#### 3D Generation Techniques for Stable Diffusion & Related Diffusion Based 3D Generation
Stable Diffusion (SD) is a powerful text-to-image generation model that has inspired the development of several techniques for generating 3D images and scenes based on text prompts. Two of the most notable methods are:

#### Text to 3D
https://dreamfusion3d.github.io/  
https://github.com/ashawkey/stable-dreamfusion  

#### DMT Meshes / Point Cloud Based
https://github.com/Firework-Games-AI-Division/dmt-meshes

#### 3D radiance Fields
not technically stable diffusion but diffusion based 3D modeling  
https://sirwyver.github.io/DiffRF/  

#### Novel View Synthesis
not technicall stable diffusion but is related  
https://3d-diffusion.github.io/  

##### NeRF Based:
This technique uses the Neural Radiance Fields (NeRF) algorithm to generate 3D models based on 2D images. The Stable Dreamfusion repository on Github (https://github.com/ashawkey/stable-dreamfusion) is an implementation of this technique for Stable Diffusion. It allows users to generate high-quality 3D models from text prompts and can be customized to achieve specific effects and styles.

##### Img to Fspy to Blender:
This technique uses a combination of image analysis and 3D modeling software to create 3D scenes based on 2D images. It involves using the Img to Fspy tool (https://fspy.io/) to analyze an image and generate a camera location, then importing the camera location into Blender to create a 3D scene. A tutorial on this technique is available on YouTube (https://youtu.be/5ntdkwAt3Uw) and provides step-by-step instructions for generating 3D scenes based on images.

Both of these techniques offer powerful tools for generating 3D images and scenes based on text prompts. They require some additional software and processing time, but the results can be impressive and add a new dimension to the images generated by Stable Diffusion.

##### Image to Shapes
3D shapes on top of images. A tutorial on this technique is available on YouTube by Albert Bozesan (https://youtu.be/ooSW5kcA6gI) and provides step-by-step instructions for building 3D shapes based on images. Roughly we lay out the image inside blender then extrude the shapes and polish the model while using the image as texture.

Similar to https://github.com/jeacom25b/blender-boundary-aligned-remesh https://www.youtube.com/watch?v=AQckQBNHRMA

#### 3D Texturing Techniques for Stable Diffusion
Stable Diffusion (SD) is a powerful text-to-image generation model that has inspired the development of several techniques for generating 3D textures based on text prompts. Two of the most notable methods are:

##### Using Stable Diffusion for 3D Texturing:
This technique involves using Stable Diffusion to generate high-quality images based on text prompts, and then using those images as textures for 3D models. This technique is described in detail in an article on 80.lv (https://80.lv/articles/using-stable-diffusion-for-3d-texturing/) and offers a powerful tool for generating realistic and detailed 3D textures.

##### Dream Textures:
This is a project on Github (https://github.com/carson-katri/dream-textures) that uses Stable Diffusion to generate high-quality textures for 3D models. It allows users to customize the texture generation process and create unique and creative textures based on text prompts.



#### Music

##### Riffusion
https://en.wikipedia.org/wiki/Riffusion

#### Image-Based Mind Reading
https://the-decoder.com/stable-diffusion-can-visualize-human-thoughts-from-mri-data/

#### Synthetic Data Creation
https://hai.stanford.edu/news/could-stable-diffusion-solve-gap-medical-imaging-data



## How it Works





## Beginner's How To





## Popular UI's


### Automatic 1111
Automatic 1111's superpower is it's rapid development speed and leveraging of community addons, usually within days of research being shown an addon for it in Auto1111 appears, if those addons prove popular enough they are eventually merged into standard features of the UI. I would likely say that because of this Aato1111 is the default choice of UI for most users until they have a specialized need or desire something easier to use. It is a powerful and comprehensive UI

Tutorials:  

Local Installation:  

Colab:  

#### Automatic 1111 Extensions
Stable Diffusion (SD) is a powerful text-to-image generation model that has inspired the development of several extensions and plugins that enhance its capabilities and offer new features. Many of these extensions can be found on the Github repository for AUTOMATIC1111 (https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions) and can be installed through the extensions tab inside of AUTOMATIC1111, or by cloning the respective Github repositories into the extensions folder inside your AUTOMATIC1111 webUI/extensions directory.  

Some of the most notable extensions for Stable Diffusion are:  

Ultimate Upscale: This is an extension that uses the ESRGAN algorithm to upscale images generated by Stable Diffusion to high-resolution versions. The Github repository for this extension is available at https://github.com/Coyote-A/ultimate-upscale-for-automatic1111, and a FAQ for the extension is available at https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ.  

Config Presets: This is an extension that allows users to save and load configuration presets for Stable Diffusion. It simplifies the process of setting up Stable Diffusion for specific tasks and allows users to switch between presets quickly. The Github repository for this extension is available at https://github.com/Zyin055/Config-Presets.  

Image Browser: This is an extension that provides a visual interface for browsing and selecting images to use as input for Stable Diffusion. It simplifies the process of selecting and managing images and allows users to preview images before generating output.  

Prompt Tag Autocomplete: This is an extension that provides autocomplete suggestions for text prompts based on previously used prompts. It speeds up the process of entering prompts and reduces the likelihood of errors.  

Txt2Mask: This is an extension that generates masks from text prompts. It allows users to select specific regions of an image to generate output from and can be useful for tasks such as object removal or image editing.  

Ultimate HD Upscaler: This is an extension that uses a neural network to upscale images generated by Stable Diffusion to high-resolution versions. It offers improved upscaling quality compared to traditional algorithms.  

Aesthetic Scorer: This is an extension that uses a neural network to score the aesthetic quality of images generated by Stable Diffusion. It can be used to evaluate the quality of generated images and provide feedback for improvement.  

Tagger: This is an extension that adds tags to generated images based on the input text prompts. It can be useful for organizing and managing large numbers of generated images.  

Inspiration Images: This is an extension that provides a database of images for use as input prompts. It can be useful for generating images based on specific themes or styles.  

These extensions offer a range of useful features and tools for working with Stable Diffusion. They can enhance the capabilities of the model and simplify the process of generating high-quality images based on text prompts.  

Depth Map Library and Poser
https://github.com/jexom/sd-webui-depth-lib 

OpenPose Editor
https://github.com/fkunn1326/openpose-editor  

### Kohya
attempts to get it to run on linux and then in a colab
https://github.com/kohya-ss/sd-webui-additional-networks

https://www.bilibili.com/read/cv21373135?from=articleDetail
https://github.com/Thund3rPat/kohya_ss-linux

#### Addons
https://github.com/kohya-ss/sd-webui-additional-networks




### EasyDiffusion / Formerly Stable Diffusion UI
https://github.com/cmdr2/stable-diffusion-ui


### InvokeAI
https://github.com/invoke-ai/InvokeAI

Unified Canvas Option

Diffusers can be used natively


### DiffusionBee (Mac OS)
https://github.com/divamgupta/diffusionbee-stable-diffusion-ui

### NKMD GUI


### ComfyUi
https://github.com/comfyanonymous/ComfyUI  

### AINodes
it's not popular yet, but I expect it will be  
https://www.reddit.com/r/StableDiffusion/comments/11psrvp/ainodes_teaser_update/
https://github.com/XmYx/ainodes-engine

## Model and Training UI's
webui model toolkit https://github.com/arenatemp/stable-diffusion-webui-model-toolkit





### Other Sofware Addons that Act like a UI
https://github.com/carson-katri/dream-textures





## Hardware Requirements and Cloud-Based Solutions



### Xformers for Stable Diffusion

Xformers is a set of transformers that can be used as an alternative to Stable Diffusion's built-in transformers for text-to-image generation. Xformers can run on fewer resources and provide comparable or better results than built-in transformers, making them a popular choice for many users.

However, Xformers can be prone to compatibility issues when upgrading, and many users have reported problems when upgrading to newer versions. Some users have had to downgrade to previous versions to resolve these issues.

To downgrade Xformers, users can follow these instructions:

Navigate to your Stable Diffusion webUI folder and go into venv, then scripts.

Select the navigation bar and type in CMD. This should open a CMD window in this folder. Alternatively, users can open the CMD window and navigate to this folder.

Type "activate" and hit enter to activate the virtual environment.

Run the following command: "pip install xformers==0.0.17.dev449".

This will downgrade Xformers to the specified version and resolve any compatibility issues. However, users should be aware that downgrading may result in some loss of functionality or performance compared to newer versions. It is recommended to carefully evaluate the specific needs and requirements of your project before downgrading.






## Resources
Stable Horde



### Communities/Sites to Use it



### Where to Get Models
Pickling and Safetensors

Pickling: This is a technique used to serialize and deserialize Python objects. Pickling is used in SD to save and load models, as well as to transfer data between processes. Pickling can be used to save the state of the model at various stages of training or to transfer a model between different machines or environments. However, pickling can also introduce security risks if used improperly, as it allows for arbitrary code execution.

SafeTensors: This is a technique used to ensure that the tensors used in SD are safe and do not pose a security risk. SafeTensors are created by wrapping tensors with metadata that defines their type and shape. This metadata can be used to verify that tensors are being used correctly and to prevent attacks such as tensor poisoning.

Both pickling and SafeTensors are important techniques for ensuring the safety and performance of SD. They allow for models to be saved, loaded, and transferred between environments, while also ensuring that the data used in SD is secure and cannot be manipulated by attackers.



### Communities/Sites to discuss and share things with it



### Sites for Prompt Inspiration for Stable Diffusion

Stable Diffusion (SD) is a powerful text-to-image generation model that relies on high-quality text prompts to generate images. To generate the best possible results, users need access to a wide range of high-quality prompts that are relevant to their projects.

Fortunately, there are several websites and platforms that offer prompt inspiration for SD, including:

Libraire.ai: This is a website that offers a wide range of writing prompts and exercises for writers. Many of these prompts can be adapted for use with SD to generate images based on text.

Lexica.art: This is a platform that offers a range of creative prompts for artists and writers. These prompts can be used to generate ideas for SD images and to refine text prompts for better results.

Krea.ai: This is a platform that offers a range of prompts for creative projects, including writing and art prompts. Many of these prompts can be adapted for use with SD to generate high-quality images based on text.

PromptHero.com: This is a website that offers a wide range of prompts for writing, storytelling, and creative projects. These prompts can be used to generate ideas for SD images and to refine text prompts for better results.

OpenArt.ai: This is a platform that offers a range of creative prompts and challenges for artists and designers. These prompts can be used to generate ideas for SD images and to refine text prompts for better results.

PageBrain.ai: This is a website that offers a range of writing prompts and exercises for writers. Many of these prompts can be adapted for use with SD to generate images based on text.

These websites offer a wealth of creative prompts and exercises that can be adapted for use with SD to generate high-quality images based on text. Users should explore these sites and other similar resources to find the best prompts for their specific projects and goals.





## Stable Diffusion (SD) Core and Models

At the core of SD is the stable diffusion model, which is contained in a ckpt file. The stable diffusion model consists of three sub-models:

Variational autoencoder (VAE): This sub-model is responsible for compressing and decompressing the image data into a smaller latent space. The VAE is used to generate a representation of the input image that can be easily manipulated by the other sub-models.

U-Net: This sub-model is responsible for performing the diffusion process that generates the final image. The U-Net is used to gradually refine the image by adding or removing noise and information based on the text prompts.

CLIP: This sub-model is responsible for guiding the diffusion process with text prompts. CLIP is a natural language processing model that is used to generate embeddings of the text prompts that are used to guide the diffusion process.

Different models can use different versions of the VAE, U-Net, and CLIP models, depending on the specific requirements of the project. In addition, different samplers can be used to perform denoising in different ways, providing additional flexibility and control over the image generation process.

Understanding the core components and models of SD is important for optimizing its performance and for selecting the appropriate models and settings for specific projects.



### Base Models for Stable Diffusion

Stable Diffusion (SD) relies on pre-trained models to generate high-quality images from text prompts. These models can be broadly categorized into two types: official models and community models.

Official models are trained on large datasets of images, typically billions of images, and are often referred to by their dataset size. For example, the LAION-2B model was trained on a dataset of 2 billion images, while the LAION-5B model was trained on a dataset of 5.6 billion images. These models are typically trained on a wide range of images and can generate high-quality images that are suitable for many different applications.

Community models, on the other hand, are models that have been finetuned by users for specific styles or objects. These models are often based on the official models, but with modifications to the Unet and decoder or just the Unet. For example, a user might finetune an official model to generate images of specific animals or to generate images with a particular style or aesthetic.

The choice of which model to use depends on the specific requirements of the project. Official models are generally more versatile and can be used for a wide range of applications, but may not produce the specific style or quality of image desired. on the other hand, community models may be more tailored to specific applications but may not be as versatile as official models.

It is important to carefully evaluate the specific needs and requirements of a project before selecting a model and to consider factors such as dataset size, style, object, and computational resources when making a decision.

#### Stable Diffusion Models 1.4 and 1.5

Stable Diffusion (SD) has gone through several iterations of models, each trained on different datasets and with different hyperparameters. The earliest models, 1.1, 1.2, and 1.3, were trained on subsets of the LAION-2B dataset at resolutions of 256x256 and 512x512.

Model 1.4 was the first SD model to really stand out, and it was trained on the LAION-aesthetics v2.5+ dataset at a resolution of 512x512 for 225k steps. Model 1.5 was also trained on the LAION-aesthetics v2.5+ dataset, but for 595k steps. It comes in two flavors: vanilla 1.5 and inpainting 1.5.

Both models are widely used in the SD community, with many finetuned models and embeddings based on 1.4. However, 1.5 is considered the dominant model in use because it produces good results and is a solid all-purpose model.

One important consideration for users is compatibility between models. Most things are compatible between 1.4 and 1.5, which makes it easier for users to switch between models and take advantage of different features or capabilities.

It is important to evaluate the specific needs and requirements of a project when selecting a model and to consider factors such as dataset size, resolution, and hyperparameters when making a decision.

#### Stable Diffusion Models 2.0 and 2.1

Stable Diffusion (SD) models 2.0 and 2.1 were released closely together, with 2.1 considered an improvement over 2.0. Both models were trained on the LAION-5B dataset, which contains roughly 5 billion images, compared to the LAION-2B dataset used for earlier models.

One of the biggest changes from a user perspective was the switch from CLIP (OpenAI) to OpenCLIP, which is an open-source version of CLIP. While this is a positive development from an open-source perspective, it does mean that some workflows and capabilities that were easy to achieve in earlier versions may not be as easy to replicate in 2.0 and 2.1.

SD2.1 comes in both 512x512 and 768x768 versions. Because it uses OpenCLIP instead of CLIP, some users have expressed frustration at not being able to replicate their SD1.5 workflows on SD2.1. However, new fine-tuned models and embeddings are emerging rapidly, which are extending the capabilities of SD2.1 and making it more versatile for different applications.

As with earlier models, it is important to carefully evaluate the specific needs and requirements of a project when selecting a model and to consider factors such as dataset size, resolution, and hyperparameters when making a decision.

##### 512-Depth Model for Image-to-Image Translation

The 512-depth model is a Stable Diffusion model that enables image-to-image translation at a resolution of 512x512. While conventional image-to-image translation methods can suffer from issues with preserving the composition of the original image, the 512-depth model is designed to preserve composition much better. However, it is important to note that this model is limited to image-to-image translation and does not support other tasks such as text-to-image generation or inpainting.



### VAE (Variational Autoencoder) in Stable Diffusion

In Stable Diffusion, the VAE (or encoder-decoder) component is responsible for compressing the input images into a smaller, latent space, which helps to reduce the VRAM requirements for the diffusion process. In practice, it is important to use a decoder that can effectively reconstruct the original image from the latent space representation.

While the default VAE models included with Stable Diffusion are suitable for many applications, there are other fine-tuned models available that may better meet specific needs. For example, the Hugging Face model repository includes a range of fine-tuned VAE models that may be useful for certain tasks.

When selecting a VAE model, it is important to consider factors such as dataset size, resolution, and other hyperparameters that may impact performance. Ultimately, the choice of VAE model will depend on the specific needs and requirements of the project at hand.

#### Original Autoencoder in Stable Diffusion

The original autoencoder included in Stable Diffusion is the default encoder-decoder used in the model. While it is generally effective at compressing images into a latent space for the diffusion process, it may not perform as well on certain types of images, particularly human faces.

Over time, several fine-tuned autoencoder models have been developed and made available to the community. These models often perform better than the original autoencoder for specific tasks and image types.

When selecting an autoencoder model for a specific application, it is important to consider factors such as image resolution, dataset size, and other hyperparameters that may impact performance. Ultimately, the choice of the autoencoder model will depend on the specific needs and requirements of the project at hand.

#### EMA VAE in Stable Diffusion

The EMA (Exponential Moving Average) VAE is a fine-tuned encoder-decoder included in Stable Diffusion that is specifically designed to perform well on human faces. This model uses an exponential moving average of the encoder weights during training, which helps to stabilize the training process and improve overall performance.

Compared to the original autoencoder included with Stable Diffusion, the EMA VAE generally produces better results on images of human faces. However, it is important to consider other factors such as image resolution, dataset size, and other hyperparameters when selecting a VAE model for a specific application.

Overall, the EMA VAE is a valuable addition to the range of encoder-decoder models available in Stable Diffusion, particularly for applications that require high-quality image generation of human faces.

#### MSE VAE in Stable Diffusion

The MSE (Mean Squared Error) VAE is another fine-tuned encoder-decoder included in Stable Diffusion that is designed to perform well on images of human faces. This model uses MSE as the reconstruction loss during training, which can help to improve the quality of the reconstructed images.

Compared to the original autoencoder and other VAE models included with Stable Diffusion, the MSE VAE generally produces better results on images of human faces. However, as with any model selection, it is important to consider other factors such as image resolution, dataset size, and other hyperparameters.

Overall, the MSE VAE is a useful option for applications that require high-quality image generation of human faces, particularly when used in combination with other techniques such as diffusion and CLIP-guidance.



### Samplers
samplers are used in Stable Diffusion to denoise images during the diffusion process. They are different methods to solve differential equations, and there are both classic methods like Euler and Heun as well as newer neural network-based methods like DDIM, DPM, and DPM2. Some samplers are faster than others, and some converge to a final image while others like ancestral samplers simply keep generating new images with an increasing number of steps. It's important to test and compare the speed and performance of different samplers for different use cases, but generally, the DPM++ sampler is considered the best option for most situations.

https://www.youtube.com/watch?v=gtr-4CUBfeQ

#### Ancestral Samplers
Ancestral samplers are designed to maintain the stochasticity of the diffusion process, where a small amount of noise is added to the image at each step, leading to different possible outcomes. This is in contrast to non-ancestral samplers, which aim to converge to a single image by minimizing diffusion loss. Ancestral samplers can produce interesting and diverse results with a low number of steps, but the downside is that the generated images can be more noisy and less realistic compared to the results obtained from non-ancestral samplers.

##### DPM++ 2S A Karras
DPM++ 2S A Karras is a two-step DPM++ solver. The "2S" in the name stands for "two-step". The "A" means it is an ancestral sampler and the "Karras" refers to the fact that it is based on the architecture used in the StyleGAN2 paper by Tero Karras et al.

##### DPM++ A
DPM++ A is an ancestral sampler version of the DPM++ sampler, meaning that it adds a little bit of noise at each step and never converges to a final image. It is a multi-step sampler that is based on a neural network approach to solving the diffusion process. It has been shown to produce high-quality results and is often used for generating images with complex textures and patterns. However, it can be computationally expensive and may take longer to generate images compared to other samplers.

##### Euler A
Euler A is an ancestral sampler that uses the classic Euler method to solve the discretized differential equations involved in the denoising process but adds a bit of noise at each step. This results in an image that is not necessarily converging to a single solution but rather keeps generating new variations at each step. Euler A is particularly effective at generating high-quality images at low step counts and offers a degree of control over the amount of noise added at each step for adjusting the output image.

##### DPM Fast
DPM Fast is a fast implementation of DPM (Dynamic Progressive Mesh) sampler, which is a neural network-based method of solving the problem of image denoising in Stable Diffusion models. It is a single-step method that is designed to converge faster than other methods, but it sacrifices some image quality to achieve this speed. DPM Fast is typically used for large batch processing, where speed is of the utmost importance. However, it may not be suitable for high-quality image generation where image fidelity is a priority.

##### DPM Adaptive
DPM Adaptive is a sampling method for Stable Diffusion that adapts the number of steps required to achieve a certain level of denoising based on the input image. It is designed to be more efficient than other methods by reducing the number of unnecessary steps and thus, the overall processing time. However, unlike other samplers, DPM Adaptive does not converge to a final image, meaning it will continue generating different variations of the image with an increasing number of steps. It is particularly useful for large images that require more processing time to denoise.

#### DPM++
DPM++ is a diffusion probabilistic model that uses a fast solver to speed up guided sampling. Compared to other samplers like Euler, LMS, PLMS, and DDIM, DPM++ is super fast and can achieve the same result in fewer steps. Its speed makes it a popular choice for generating high-quality images quickly. The DPM++ model is described in two research papers, available at the links provided.
PAPER: https://arxiv.org/pdf/2211.01095.pdf
PAPER: https://arxiv.org/pdf/2206.00364.pdf

##### DPM++ SDE
DPM++ SDE is a stochastic version of the DPM++ sampler. It solves the diffusion process using a stochastic differential equation (SDE) solver, which can handle both continuous and discrete-time noise. This sampler is designed to handle larger-scale guided sampling and can generate high-quality images in a relatively small number of steps. It is also one of the fastest DPM++ samplers available. The Karras version is a similar sampler that produces similar images but is optimized for smaller guidance scales.

##### DPM++ 2M
DPM++ 2M is a multi-step sampler based on the Diffusion Probabilistic Models (DPM++) solver. It is designed to perform better for large guidance scales and produces high-quality images in fewer steps compared to other samplers. The Karras version is also available, which produces similar results to the original DPM++ 2M sampler. DPM++ 2M is recommended for users who want to generate high-quality images with large guidance scales efficiently.



### Community Models
#### Fine Tuned
Fine-tuned models for Stable Diffusion are models that have been trained on top of the pre-trained Stable Diffusion model using a specific dataset or a specific task. These fine-tuned models can be more specialized and provide better results for certain tasks, such as generating images of specific objects or styles.

For example, a fine-tuned model for generating anime-style images can be trained on a dataset of anime images. Similarly, a fine-tuned model for generating high-resolution images can be trained on a dataset of high-resolution images.

Fine-tuned models can be created using transfer learning, where the pre-trained model is used as a starting point and the weights are fine-tuned on the specific task or dataset. This approach can significantly reduce the time and resources required to train a new model from scratch.

There are many fine-tuned models available for Stable Diffusion, and they can be found on various repositories and platforms, such as Hugging Face, GitHub, and other online communities.

#### Merged/Merges
In Stable Diffusion, merged models are created by combining the weights of two or more pre-trained models to create a new model. This process involves taking the learned parameters of each model and averaging them to create a new set of weights.

Merging models is often done to combine the strengths of multiple models and create a new model that is better suited for a specific task. For example, one might merge a model that is good at generating realistic faces with a model that excels at generating landscapes to create a new model that can generate realistic faces in landscapes.

Merging models requires some knowledge of deep learning and neural networks, as the models being merged need to have similar architectures and be trained on similar tasks to be effectively combined. However, there are many pre-trained models available in Stable Diffusion that have already been merged and fine-tuned for specific tasks, making it easier for users to quickly find and use models that are suitable for their needs.

##### Tutorial for Add Difference Method 
An alternative method to merge models is the use of the merge_lora script by kohya_ss.

To use this method, first, create a mix of the target model and the LoRa model using the merge_lora script. The resulting image is almost identical to just adding the LoRa, with the difference attributed to small rounding errors.

Next, add the LoRa to the target model, and also add the result of the add_difference method applied to the fine-tuned model and the mix of the target model and LoRa. The resulting merge, called the Ultimate_Merge, is 99.99% similar to the target model and can handle massive merges of hundreds of specialized models with the preferred mix without affecting it much. The Ultimate_Merge only loses 0.01% or even less of the information.

link to original tutorial/comment (NSFW) https://www.reddit.com/r/sdnsfw/comments/10nb2jr/comment/j67trgn/



#### Megamerged/MegaMerges
Megamerged models in Stable Diffusion are models that have been created by merging more than 5 models with a specific style, object, or capabilities in mind. These models can be quite complex and powerful and are often used for specific purposes or applications.

Creating a megamerged model involves taking several existing models and merging them together in a way that preserves the desired features of each individual model. This can be done using techniques like add_difference or merge_lora, as well as other methods. The resulting megamerged model is a new model that combines the strengths of each of the individual models that were used to create it.

Megamerged models can be quite powerful and effective, but they can also be more complex and difficult to work with than simpler models. They may require more VRAM and longer training times, and they may require more expertise to fine-tune and optimize for specific tasks. However, for certain applications and use cases, megamerged models can be an effective tool for achieving high-quality results.

#### Embeddings
Embeddings in Stable Diffusion are a way to add additional information to the model through text prompts. Community embeddings are created through textual inversion and can be added to prompts to achieve a desired style or object without using a fully fine-tuned model. These embeddings are not a checkpoint, but rather a new set of embeddings created by the community. Using embeddings can improve the quality and specificity of the generated images. Embeddings can be used to reduce biases within the original model or mimic visual styles.

#### Community Forks
Style2Paints
Community forks are variations of the Stable Diffusion model that are developed and maintained by individuals or groups within the community. One such fork is Style2Paints, which is focused on being more of an artist's assistant than creating random generations. It seems to be highly anime-focused, but it is doing some interesting things with sketch infilling. The Style2Paints fork can be found on GitHub and includes a preview of version 5.
https://github.com/lllyasviel/style2paints/tree/master/V5_preview 





## Capturing Concepts / Training
Capturing concepts involves training a model to generate images that match a certain style or object. This can be done in several ways, such as using a dataset of images that represent the desired style or object, or by fine-tuning an existing model on a small dataset of images that match the desired concept.

One approach to capturing concepts is to use a method called "guided diffusion," which involves generating images that match a given prompt or text description. This can be done by using a pre-trained model and fine-tuning it on a small dataset of images that match the desired concept, or by using a style transfer method to transfer the desired style onto a set of images.

Another approach is to use a method called "latent space interpolation," which involves exploring the latent space of a pre-trained model and manipulating the latent vectors to generate images that match a desired style or object. This method can be used to generate new images that are similar to a given image or to explore the space of different styles or objects.

Overall, capturing concepts involves training a model to generate images that match a desired style or object, and there are several methods available for doing so, including guided diffusion and latent space interpolation.



### Image2Text
Image2text is a technique used to convert images into text descriptions, also known as image captioning. It involves using a trained model to generate a textual description of the content of an image. This can be useful for a variety of applications, such as generating captions for social media posts or providing context for image datasets used in machine learning.

There are a few different approaches to image captioning, such as using a CNN-RNN model, which involves using a convolutional neural network to extract features from an image and then passing those features to a recurrent neural network to generate a description. Other models may use attention mechanisms or transformer architectures.

To train an image captioning model, a large dataset of images with corresponding text descriptions is typically used. The model is then trained on this dataset using a loss function that compares the generated captions to the actual captions. Once trained, the model can be used to generate captions for new images.

In the context of mixing two concepts, image2text can be used to generate textual descriptions of the different styles or objects being combined. These descriptions can then be used as prompts for a diffusion model to generate an image that combines those concepts.

#### CLIP Interrogation
CLIP Interrogator is a Python package that enables users to find the most suitable text prompts that describe an existing image based on the CLIP model. This tool can be useful for generating and refining prompts for image generation models or for labeling images programmatically during training.

CLIP Interrogator is available on GitHub and can be installed via pip. The package also includes a demo notebook showcasing the tool's functionality. Additionally, the package can be used with the Hugging Face Transformers library to further streamline the prompt generation process.
https://github.com/pharmapsychotic/clip-interrogator
DEMO: https://huggingface.co/spaces/pharma/CLIP-Interrogator

#### BLIP Captioning
BLIP (Bootstrapping Language-Image Pre-training) is a framework for pre-training vision and language models that can generate captions for images. It uses a two-stage approach, where the first stage involves training an image encoder and a text decoder on large-scale image-caption datasets, and the second stage involves fine-tuning the model on a smaller dataset with captions and corresponding prompts. This fine-tuning process uses a novel method called Contrastive Learning for Prompt (CLP) which aims to learn the relationship between the image and the prompt.

BLIP Image Captioning allows you to generate prompts for an existing image by interrogating the model. This is helpful in crafting your own prompts or for programatically labeling images during training. BLIP2 is the latest version of BLIP which has been further improved with new training techniques and larger datasets. A demo of BLIP Image Captioning can be found on the Hugging Face website.
Paper: https://arxiv.org/pdf/2201.12086.pdf
Summary: https://ahmed-sabir.medium.com/paper-summary-blip-bootstrapping-language-image-pre-training-for-unified-vision-language-c1df6f6c9166
DEMO: https://huggingface.co/spaces/Salesforce/BLIP
BLIP2: 

#### DanBooru Tags / Deepdanbooru
Danbooru is a popular anime and manga imageboard website where users can upload and tag images. DeepDanbooru is a neural network trained on the Danbooru2018 dataset to automatically tag images with relevant tags. The tags can then be used as prompts to generate images in a particular style or with certain objects.

DeepDanbooru is available as a web service or can be run locally on a machine with GPU support. The DeepDanbooru model is trained on more than 3 million images and over 10,000 tags, and is capable of tagging images with a high degree of accuracy.

Using DeepDanbooru tags as prompts can be a powerful tool for generating anime and manga-style images or images featuring particular characters or objects. It can also be used for automating the tagging process for large collections of images.

#### Waifu Diffusion 1.4 tagger - Using DeepDanBooru Tags
Waifu Diffusion 1.4 tagger is a tool developed using the DeepDanBooru tagger to automatically generate tags for images. The tool uses Stable Diffusion 1.4 model to generate images and DeepDanBooru model to generate tags. The generated tags can be used for various purposes such as organizing and searching images.

The tagger works by taking an input image and generating tags for it using DeepDanBooru model. The generated tags are then displayed alongside the image. The user can edit the generated tags and add new tags as required. Once the tags are finalized, they can be saved and used for organizing and searching images.

The tool is available as an open-source project on GitHub and can be used by anyone for free.
https://github.com/toriato/stable-diffusion-webui-wd14-tagger



### Dataset and Image Preparation
Dataset and Image Preparation is a crucial step in training and generating images with stable diffusion models. A well-prepared dataset can lead to better image quality and more efficient training.

Image preparation is also important to ensure that images are of good quality and uniform in size. Images can be resized and cropped to a consistent aspect ratio, and color correction can be applied to ensure consistency across the dataset.

A screenshot pipeline can be used to automatically extract screenshots from anime or video game footage. This can be a more efficient way to gather images for training or generating images in a specific style.

Overall, preparing a high-quality dataset is essential for stable diffusion models to generate high-quality images.

Tutorial: https://github.com/nitrosocke/dreambooth-training-guide

Screenshot Pipeline: https://github.com/cyber-meow/anime_screenshot_pipeline

#### Captioning
Captioning is the process of providing textual descriptions or labels to images, which is a crucial step in many machine-learning tasks, such as image recognition, object detection, and image captioning. In the context of training Stable Diffusion models, captioning can be helpful in providing additional context and guidance to the model, particularly when dealing with images of specific objects or styles.

For example, when training a model to generate images of a particular character with different hairstyles or clothing, providing captions that mention the character's hair or clothing can help the model to focus on remembering the character's other built-in features and reproduce these features more consistently while allowing for variation of the features that were captioned.

Captioning can also be useful in creating training datasets by automatically generating captions for images using techniques like object recognition or text-based image retrieval. These captions can then be used to train models for a variety of image-related tasks, including Stable Diffusion.

#### Regularization/Classifier Images
Regularization/Classifier Images are images used during the training process to help stabilize and regularize the model. They are typically created by training a classifier on a set of images and using the activations of that classifier as a form of regularization during training.

The use of regularization images was initially met with skepticism in the Stable Diffusion community but has since been shown to be effective in improving model stability and image quality.

The process of creating regularization images involves training a classifier on a dataset of images and then using the activations of that classifier as a form of regularization during the training process. This helps to ensure that the model is not overfitting to the training data and is able to generalize to new images.

In addition to their use in regularization, classifier images can also be used to generate prompts for image generation. By identifying the features and attributes of images that are most important for classification, these images can be used to guide the generation of new images that meet certain criteria.

Overall, regularization/classifier images are an important tool in the stable diffusion training process, helping to ensure that models are stable, generalizable, and capable of generating high-quality images.
https://www.reddit.com/r/StableDiffusion/comments/z9g46h/i_was_wrong_classifierregularization_images_do/



### Training
Training is the process of fine-tuning a pre-existing model or creating a new one from scratch to generate images based on a specific subject or style. This is achieved by feeding the model with a large dataset of images that represent the subject or style. The model then learns the patterns and features of the input images and uses them to generate new images that are similar in style or subject.

Training a model can be done in various ways, including transfer learning, where a pre-existing model is fine-tuned on a new dataset, or by creating a new model from scratch. The process typically involves setting hyperparameters, selecting the training dataset, defining the loss function, and training the model using an optimizer.

Once a model is trained, it can be used to generate new images that represent the subject or style it was trained on. This can be useful for creating custom art, generating images for specific applications, or even creating new datasets for further training. Training a model can be a complex and time-consuming process, but it can also be very rewarding in terms of the results that can be achieved.

#### File Type Overview
Most common files types used as models or embeddings

Models:
.ckpt (Checkpoint file): This is a file format used by TensorFlow to save model checkpoints. It contains the weights and biases of the trained model and can be used to restore the model at a later time.
.safetensor (SafeTensor file): This is a custom file format used by Stable Diffusion to store models and embeddings. It is optimized for efficient storage and retrieval of large models and embeddings, is also designed to be more secure. 
.pth (PyTorch model file): This is a file format used by PyTorch to save trained models. It contains the model architecture and the learned parameters.
.pkl: A Python pickle file, which is a serialized object that can be saved to disk and loaded later. This is the most common file type for saved models in Stable Diffusion.
.pt: A PyTorch model file, which is used to save PyTorch models. This file type is also used in Stable Diffusion for saved models.
.h5: A Hierarchical Data Format file, which is commonly used in machine learning for saving models. This file type is used for some Stable Diffusion models.

Embeddings:
.pt: PyTorch tensor file, which is a file format used for PyTorch tensors. This is the most common file type for embeddings in Stable Diffusion.
.npy: NumPy array file, which is a file format used for NumPy arrays. Some Stable Diffusion embeddings are saved in this format.
.h5: Hierarchical Data Format file, which can also be used for saving embeddings in Stable Diffusion.
.bin (Binary file): This is a general-purpose file format that can be used to store binary data, including models and embeddings. It is a compact format that is efficient for storing large amounts of data.

#### Textual Inversion
Textual inversion is a technique in which a new keyword is created to represent data that is already known to the model, without changing its weights. It can be particularly useful for creating images of characters or people. Textual inversion can be used in conjunction with almost any other option and can help achieve more consistent results when training models. It is not simply a compilation of prompts, but rather a way to push the output toward a desired outcome. By mixing and matching different techniques, interesting and unique results can be achieved.

Textual inversion is trained on a model so although it will often work with compatible models this is not always the case. 

https://github.com/rinongal/textual_inversion
COLAB: https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion_textual_inversion_library_navigator.ipynb

Train New Embedding Tutorial: https://youtu.be/7OnZ_I5dYgw

##### Negative Embedding
A negative embedding is an embedding used as a negative prompt to avoid certain unwanted aspects in generated images. These embeddings are typically created by generating images using only negative prompts. They can be used to group or condense a long negative prompt into a single word or phrase. Negative embeddings are useful in improving the consistency and quality of generated images, particularly in avoiding undesirable artistic aspects.

#### Hypernetworks
Hypernetworks are a machine learning technique that allows for the training of a model without altering its weights. This technique involves the use of a separate small network, known as a hypernetwork, to modify the generated images after they have been created. This approach can be useful for fine-tuning generated images without changing the underlying model architecture.

How Hypernetworks Work:

Hypernetworks are typically applied to various points within a larger neural network. This allows them to steer results in a particular direction, such as imitating the art style of a specific artist, even if the artist is not recognized by the original model. Hypernetworks work by finding key areas of importance in the image, such as hair and eyes, and then patching these areas in secondary latent space.

Benefits of Hypernetworks:

One of the main benefits of hypernetworks is that they can be used to fine-tune generated images without changing the underlying model architecture. This can be useful in situations where changing the model architecture is not feasible or desirable. Additionally, hypernetworks are known for their lower hardware requirements compared to other training methods.

Limitations of Hypernetworks:

Despite their benefits, hypernetworks can be difficult to train effectively. Many users have voiced that hypernetworks work best with styles rather than faces or characters. This means that hypernetworks may not be suitable for all types of image-generation tasks.

Tutorial: https://www.youtube.com/watch?v=1mEggRgRgfg

G.A.?

#### LORA
LORA, or Low-Rank Adaptation, is a technique for training a model to a specific subject or style. LORA is advantageous over Dreambooth in that it only requires 6GB of VRAM to run and produces two small files of 6MB, making it less hardware-intensive. However, it is less flexible than Dreambooth and primarily focuses on faces. LORA can be thought of as injecting a part of a model and teaching it new concepts, making it a powerful tool for fine-tuning generated images without altering the underlying model architecture. One of the primary benefits of LORA is that it has a lower hardware requirement to train, although it can be more complex to train than other techniques. It also does not water down the model in the same way that merging models does.

Training LORA requires following a specific set of instructions, which can be found in various tutorials available online. It is important to consider the weight of the LORA during training, with a recommended weight range of 0.5 to 0.7.

LORA is not solely used in Stable Diffusion and is used in other machine learning projects as well. Additionally, DIM-Networks can be used in conjunction with LORA to further enhance training.

https://github.com/cloneofsimo/lora
DEMO - Broken?: https://huggingface.co/spaces/ysharma/Low-rank-Adaptation

Training LORA
Tutorial: https://www.reddit.com/r/StableDiffusion/comments/111mhsl/lora_training_guide_version_20_i_added_multiple/?utm_source=share&utm_medium=web2x&context=3

Changing Lora Weight example: 0.5-:0.7


Number of Images in training data

Converting  Checkpoint to LORA

##### LoHa
Seems to be a LORA that has something to do with federated learning, meaning can be trained in small pieces by many computers instead of all at once in one large go? I'm not completely sure yet
Github: https://github.com/KohakuBlueleaf/LyCORIS
Paper: https://openreview.net/pdf?id=d71n4ftoCBy

#### Aescetic Gradients
Aesthetic gradients are a type of image input that can be used as an alternative to textual prompts. They are useful when trying to generate an image that is difficult to describe in words, allowing for a more intuitive approach to image generation. However, some users have reported underwhelming results when using aesthetic gradients as input. The settings to modify weight may be unclear and unintuitive, making experimentation necessary. Aesthetic gradients may work best as a supplement to a trained model, as both the model and the gradients have been trained on the same data, allowing for added variation in generated images.



### Fine Tuning / Checkpoints/Diffusers/Safetensors
To fine-tune a model, you start with a pre-trained checkpoint or diffuser and then continue training it on your own dataset or with your own prompts. This allows you to customize the model to better fit your specific needs. Checkpoints are saved models that can be loaded to continue training or to generate images. Diffusers, on the other hand, are used for guiding the diffusion process during image generation.

Fine-tuning can be done on a variety of pre-trained models, including the base models such as 1.4, 1.5, 2.0, 2.1, as well as custom models. Fine-tuning can be useful for training a model to recognize a specific subject or style, or for improving the performance of a model on a specific task.

A diffuser, checkpoint (ckpt), and safetensor are all related to the process of training and using neural network models, but they serve different purposes:

A diffuser is a term used in the Stable Diffusion framework to refer to a specific type of image generation model. Diffusers are trained using a diffusion process that gradually adds noise to an image, allowing the model to generate increasingly complex images over time. Diffusers are a key component of the Stable Diffusion framework and are used to generate high-quality images based on textual prompts.

A checkpoint (ckpt) is a file that contains the trained parameters (weights) of a neural network model at a particular point in the training process. Checkpoints are typically used for saving the progress of a training session so that it can be resumed later, or for transferring a pre-trained model to another computer or environment. Checkpoints can also be used to fine-tune a pre-trained model on a new dataset or task.

A safetensor is a file format used to store the trained parameters (weights) of a neural network model in a way that is optimized for fast and efficient loading and processing. Safetensors are similar to checkpoints in that they store the model parameters, but they are specifically designed for use with the TensorFlow machine learning library. Safetensors can be used to save and load pre-trained models in TensorFlow, and can also be used for fine-tuning or transfer learning.

In summary, diffusers are a type of image generation model used in the Stable Diffusion framework, while checkpoints and safetensors are file formats used to store and load the trained parameters of a neural network model. Checkpoints and safetensors are often used for fine-tuning or transfer learning, while diffusers are used for generating high-quality images based on textual prompts.

#### Token Based
Token-based fine-tuning is a simplified form of fine-tuning that requires fewer images and utilizes a single token to modify the model. This approach does not require captions for each image, making it easier to execute and reducing the chances of error. The single token is used to modify the model's weights to achieve the desired outcome. While token-based fine-tuning is a simpler method, it may not provide the same level of accuracy and customization as other forms of fine-tuning that use more detailed captions or multiple tokens.

##### Dreambooth
Dreambooth is a tool that allows you to fine-tune a Stable Diffusion checkpoint based on a single keyword that represents all of your images, for example, "mycat." This approach does not require you to caption each individual image, which can save time and effort. To use Dreambooth, you need to prepare at least 20 images in a square format of 512x512 or 768x768 and fine-tune the Stable Diffusion checkpoint on them. This process requires a significant amount of VRAM, typically above 15GB, and will produce a file ranging from 2GB to 5GB. Accumulative stacking is also possible in Dreambooth, which involves consecutive training while maintaining the structure of the models. However, this technique is challenging to execute. Overall, Dreambooth can be a useful tool for fine-tuning a Stable Diffusion checkpoint to a specific set of images using a single keyword.

PAPER: https://dreambooth.github.io/
TUTORIAL: https://www.youtube.com/watch?v=7m__xadX0z0 or https://www.youtube.com/watch?v=Bdl-jWR3Ukc
COLAB: https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb

##### Custom Diffusion by Adobe
Custom Diffusion by Adobe is a technique for fine-tuning a Stable Diffusion model to a specific dataset. This approach involves training a new model on the dataset using the Diffusion process, which can take several days or even weeks depending on the size and complexity of the dataset. The resulting model can then be used to generate images with the specific style or content of the training dataset.

One of the key benefits of Custom Diffusion by Adobe is its ability to generate high-quality images that are visually consistent with the training data. This makes it a powerful tool for a wide range of applications, from generating art and design to creating realistic simulations for video games and movies.

However, Custom Diffusion by Adobe is also a computationally intensive process that requires significant resources, including powerful hardware and access to large amounts of data. As such, it may not be practical for all users or applications. Additionally, the technique may require significant expertise and training to use effectively, making it more suitable for advanced users with experience in machine learning and computer vision.
https://github.com/adobe-research/custom-diffusion
https://huggingface.co/spaces/nupurkmr9/custom-diffusion

#### Caption Based Fine Tuning
Caption-based fine-tuning is a method of fine-tuning a stable diffusion model that requires a large number of images, typically in the range of hundreds to thousands. In this approach, the image captions are used as the basis for fine-tuning, allowing for multi-concept training. While this method allows for more flexibility in training, it requires more work than other methods such as token-based fine-tuning. The key advantage of this approach is its ability to capture multiple concepts in the fine-tuning process, enabling more nuanced image generation.

Caption-based fine-tuning requires a lot of captions, not necessarily a lot of images. It can be done with a smaller set of images, as long as they have a diverse range of captions that represent the desired concepts or styles.

##### Fine Tuning
Fine tuning is a technique used to create a new checkpoint based on image captions. Unlike token-based fine tuning, this method requires a lot of images, ranging from hundreds to thousands. With fine tuning, you can choose to tune just the Unet or both the Unet and the decoder. This process requires a minimum of 15GB VRAM and produces a file ranging from 2GB to 5GB in size. While conventional dreambooth codes can be used for fine tuning, it is important to select the options that allow the use of captions instead of tokens.

TUTORIAL: https://docs.google.com/document/d/1x9B08tMeAxdg87iuc3G4TQZeRv8YmV4tAcb-irTjuwc/edit
https://github.com/victorchall/EveryDream-trainer 
https://github.com/victorchall/EveryDream2trainer
https://github.com/devilismyfriend/StableTuner

#### Decoding Checkpoints
Decoding checkpoints refer to a method of using pre-trained models to generate images based on textual prompts or other inputs. These checkpoints contain a set of weights that have been optimized during the training process to produce high-quality images. The decoding process involves feeding a textual prompt into the model and using the learned weights to generate an image that matches the input. These checkpoints can be used for a wide variety of image generation tasks, including creating artwork, generating realistic photographs, or creating new designs for products. Different types of decoding checkpoints may be used for different types of tasks, and users may experiment with different models to find the one that works best for their specific needs. Overall, decoding checkpoints are a powerful tool for generating high-quality images quickly and efficiently.



### Mixing
Mixing in Stable Diffusion refers to combining different models, embeddings, prompts, or other inputs to generate novel and varied images. Image2text is a tool that can be used to analyze existing images and generate prompts that capture the style or content of the image. These prompts can then be used to generate new images using Stable Diffusion models. Additionally, mixing can be achieved by combining different models or embeddings together, either through merging or using hypernetworks. This can allow for greater flexibility in generating images with unique styles and content.

#### Using Multiple types of models and embeddings
Using multiple types of models and embeddings such as hypernetworks, embeddings, or LORA can be useful for mixing different styles and objects together. By combining the strengths of multiple models, you can create more unique and diverse images. For example, using multiple embeddings can give you a wider range of prompts to use in image generation, while combining hypernetworks can help fine-tune the generated images without changing the underlying model architecture. However, using too many models at once can lead to decreased performance and longer training times. It is important to find a balance between using multiple models and keeping your system resources efficient.

##### Multiple Embeddings
When using Stable Diffusion for image generation, it is possible to use multiple embeddings simultaneously by adding the different keywords of the embeddings to your prompt. This can be helpful when attempting to mix different styles or objects together in your generated image. By using multiple embeddings, you can create more complex and nuanced prompts for the model to generate images from.

##### Multiple Hypernetworks
Using multiple hypernetworks can help mix styles and objects together in image generation. These hypernetworks can be added to the model to modify images in a certain way after they are created, without changing the underlying model architecture. While powerful, hypernetworks can be difficult to train and require a lower hardware requirement than fine-tuning models. By using multiple hypernetworks, users can achieve more diverse and nuanced results in their image generation.
https://github.com/antis0007/sd-webui-multiple-hypernetworks

##### Multiple LORA's
To achieve a more customized image output, multiple LORA models can be used in combination with custom models and embeddings. However, some users have reported that using more than 5 LORA models simultaneously can lead to poor results. It is important to experiment with different combinations and find the optimal balance of LORA models to achieve the desired output.

#### Merging
Merging checkpoints allows for mixing two concepts together. This can be done by combining the weights of two or more pre-trained models. However, it is important to note that merging can cause a loss or weakening of some concepts in the final output due to the differences in the underlying architectures and training data of the models being merged. It is recommended to experiment with different merging approaches and models to achieve the desired results.

##### Merging Checkpoints
Merging checkpoints is a technique used to combine two different models to create a new model with characteristics of both. This process allows you to mix the models together in various proportions, ranging from 0% to 100%. By merging models, you can create entirely new styles and outputs that wouldn't be possible with a single model. However, it's important to note that merging models can also result in a loss or weakening of certain concepts. Therefore, it's important to experiment with different combinations and proportions to achieve the desired result.

#### Converting
Converting Checkpoints to LORA and Safetensors involves transforming the trained model weights into a compressed format that can be used in other applications.

To convert a checkpoint to LORA, you can use the "compress.py" script provided in the LORA repository. This script takes a trained checkpoint and compresses it into a LORA file, which can be used in other machine learning projects. This can also be done with Kohya Ui. 

To convert a checkpoint to a Safetensor, you can use the "export.py" script provided in the Safetensor repository. This script takes a trained checkpoint and exports it as a Safetensor, which is a compressed and encrypted version of the model that can be safely shared with others. This can also be done with most UIs. 

Converting checkpoints to LORA or Safetensors can be useful for sharing models with others or for using them in other applications that require compressed model files.



### One Shot Learning & Similar
One-shot learning is a machine learning technique where a model is trained on a small set of examples to classify new examples. In the context of Stable Diffusion, one-shot learning can be used to quickly train a model on a new concept or object with just a few images.

One way to do this is to use a technique called fine-tuning, where a pre-trained model is modified to fit the new data. For example, if you want to train a model to generate images of your pet cat, you can fine-tune an existing Stable Diffusion model on a small set of images of your cat. This will allow the model to learn the specific characteristics of your cat and generate new images of it.

Another approach is to use a technique called contrastive learning, where a model is trained to differentiate between positive and negative examples of a concept. For example, you can train a model to recognize your cat by showing it a few positive examples of your cat, and many negative examples of other cats or animals. This will allow the model to learn the unique features of your cat and distinguish it from other animals.

One-shot learning can be useful in scenarios where there are only a few examples of a concept, or where collecting large amounts of data is not feasible. However, it may not always produce the same level of accuracy as traditional training methods that use large datasets. Additionally, the quality of the generated images may depend on the quality of the initial few examples used for training.

#### DreamArtist (WebUI Extension)
DreamArtist is a web extension that allows users to generate custom art using Stable Diffusion. The extension provides a user-friendly interface that makes it easy for anyone to generate images without any coding experience. Users can upload their images, choose a specific style or subject, adjust settings such as resolution and noise level, and generate new images with a single click. DreamArtist also allows users to save and share their creations with others. It is a convenient tool for anyone who wants to experiment with Stable Diffusion and create unique digital art.
https://github.com/7eu7d7/DreamArtist-sd-webui-extension

#### Universal Guided Diffusion
Universal Guided Diffusion is a method for training a diffusion model that can generate diverse high-quality images from a wide range of natural image distributions. It involves conditioning the diffusion process on a universal latent code that captures global properties of the image distribution, as well as a guided conditioning signal that captures local details. This approach allows for a high degree of flexibility in generating images with diverse styles and content, making it suitable for a wide range of image-generation tasks. The code is available on GitHub, and a paper describing the method is available on arXiv.
https://github.com/arpitbansal297/Universal-Guided-Diffusion
PAPER: https://arxiv.org/abs/2302.07121





## Initiating Composition
Initiating composition in the context of stable diffusion generally refers to the process of generating an image from scratch using a combination of textual prompts and/or image inputs. This process can be done using various techniques such as fine-tuning pre-trained models, using multiple embeddings, hypernetworks, and LORAs, merging models, and utilizing aesthetic gradients. The goal is to generate an image that reflects the desired style, subject, or concept that the user has in mind. Once an image has been generated, it can be further refined and tweaked using techniques such as image manipulation, denoising, and interpolation to achieve the desired outcome.


### Text2Image
Stable Diffusion is a machine learning framework that is used for generating images from textual prompts. This is achieved through a process known as Text2Image, where textual input is used to generate corresponding images. The core functionality of Stable Diffusion is based on the use of a diffusion process, where a series of random noise vectors are iteratively modified to generate high-quality images. This process involves using a series of convolutional neural networks and other machine-learning techniques to generate the final image output.

The Text2Image functionality of Stable Diffusion has been detailed in a paper available on arXiv, and there are also various tutorials and videos available to help users understand how the framework works. The main advantage of using Stable Diffusion for generating images from text is that it can produce high-quality, realistic images with relatively little input. This makes it a useful tool for a wide range of applications, from generating art to creating realistic simulations for computer games and other applications.
Paper: https://arxiv.org/pdf/2112.10752.pdf
How SD Works: https://www.youtube.com/watch?v=1CIpzeNxIhU

#### Notes on Resolution
the initial dataset was trained on 512x512px images, so when one deviates from that size it can sometimes act like it's generating and merging two images, this is the usual culprit when an image has a double head (stacked on top of another head). Other models like 2.0 have been trained on a larger subset of 768x768 and some custom user models also have custom image size training data. The most important thing to note is that deviating from the size it is trained on can sometimes cause unforeseen strangeness in the images generated. 

#### Prompt Editing
Prompt editing is a powerful tool in Stable Diffusion that allows users to manipulate and refine prompts to guide the generation process. Prompts come in two types: positive prompts and negative prompts. Positive prompts encourage the model to generate specific features, while negative prompts discourage the model from generating unwanted features. Prompt editing techniques include prompt emphasis, which allows users to highlight specific words or phrases in the prompt, and prompt delay, which introduces a time delay between each word in the prompt to allow for more fine-tuned control over the generation process. Other techniques include alternating words and using prompts that contain specific features, such as the rule of thirds, contrasting colors, sharp focus, and intricate details. These prompt editing techniques can help users achieve more precise and nuanced control over the generated images.

https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#prompt-editing 

prompt engineering resources https://www.reddit.com/r/StableDiffusion/comments/xcrm4d/useful_prompt_engineering_tools_and_resources/?utm_source=share&utm_medium=web2x&context=3

#### Negative Prompts
Negative prompts are used to guide Stable Diffusion models away from certain image characteristics. However, the impact of negative prompts can be unpredictable and requires experimentation. It is important to note that there is no guaranteed set of negative prompts that will always produce the desired outcome, and the effectiveness of negative prompts can vary depending on the specific model, textural inversions, hypernetworks, or LoRA being used. It is recommended to focus on negative prompts that are relevant to the specific image you are trying to generate, rather than including irrelevant or meaningless prompts. Ultimately, it is important to experiment with different prompts and learn what works best for each specific use case.

#### Alternating Words
Alternating Words is a feature in Auto1111 that allows users to alternate between two keywords at each time step. This feature can be used by specifying the two keywords in square brackets separated by a vertical bar, such as [Salvador Dali|Pixel Art]. The model will then alternate between these two keywords when generating the image. This can be useful for exploring different styles or concepts in the generated images, as well as adding variety to the output.
https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#alternating-words 

#### Prompt Delay
Prompt Delay is a feature that can be used in various Stable Diffusion interfaces, allowing users to delay the appearance of certain keywords until a minimum number of steps have been reached. The syntax for Prompt Delay involves adding a delay value to the end of a keyword, represented as a decimal between 0 and 1. For example, the prompt [Salvador Dali: Pixel Art:0.2] would delay the appearance of "Pixel Art" until 20% of the process has been completed, with "Salvador Dali" being used for the remaining 80%. This feature can be useful for fine-tuning the progression and appearance of keywords in the generation process.

#### Prompt Weighting
Prompt Weighting can be used in several interfaces for stable diffusion. The syntax is [Salvador Dali:1.1 Pixel Art:1], where here Salvador Dali has a weight of 1.1 and Pixel Art has a weight of 1. The weights allow you to adjust the importance of each keyword in the prompt, with higher weights indicating more importance.

#### Ui specific Syntax
Stable Diffusion offers various UI frontends, each with its own unique or specific syntax for prompting. Examples of these syntaxes are provided with links to demonstrate the differences.


### Exploring
Exploring the latent space in Stable Diffusion can be a daunting task due to its sheer size. However, there are several methods to explore it and find the desired image. One approach is to use brute force on a small part of the space near the optimal solution. Another method is to use random words or parameters to explore the space and discover new and interesting images. Overall, exploring the latent space is a key component of using Stable Diffusion effectively, and there are various techniques available to help with this task.

#### Randomness
Using randomness in the prompts and parameters is a powerful tool to explore different styles and types of images in Stable Diffusion. Randomness can be introduced in various ways, such as through the use of random words or phrases in the prompt, or through random adjustments to parameters such as temperature or noise. This approach can lead to surprising and creative results, but can also be unpredictable and may require experimentation to achieve the desired outcome. Overall, incorporating randomness into the Stable Diffusion process can be a useful way to expand the range of possible images and generate novel and unexpected results.

##### Random Words
Random Words are a technique used in Stable Diffusion to explore different styles and types of images. The approach involves generating a large number of images using a combination of words that are randomly chosen. This technique can help to uncover new and interesting combinations of keywords and produce unique and unexpected results. There are several tools and libraries available for generating random words and incorporating them into Stable Diffusion prompts. One example is the sd-dynamic-prompts library available on GitHub.https://github.com/adieyal/sd-dynamic-prompts 

##### Wildcards
Wildcards are a feature in Stable Diffusion that allow users to explore the latent space by using a combination of random words and placeholders. These placeholders, represented by asterisks (*), can be used to substitute for any word or phrase, allowing for greater flexibility in the prompts. Users can generate a large number of images with randomly chosen words and placeholders, allowing for a wider exploration of the model's capabilities. This feature is available in the Stable Diffusion AI Prompt Examples repository on GitHub.
https://github.com/joetech/stable-diffusion-ai-prompt-examples

#### Brute Force
Brute force is a method of exploring the parameter space systematically. It can be performed in one, two, or multiple dimensions, such as exploring the impact of the configuration scale, steps, samplers, denoising strength, etc. This approach involves systematically testing all possible combinations of parameters to find the optimal solution or to explore the parameter space. However, brute force can be computationally expensive and time-consuming, especially when exploring high-dimensional spaces. As such, it is important to carefully consider the trade-off between computational cost and the potential benefits of using brute force.

##### Prompt Matrix
A prompt matrix is a method of generating a grid of images by combining two prompts to create all possible combinations. For example, if you have two prompts "chaotic" and "evil," a prompt matrix would generate a grid of images showing all possible combinations of the two prompts such as "chaotic good," "chaotic neutral," "evil good," and so on. This technique can be useful for exploring different combinations of prompts and generating a wide range of images.

##### XY Grid
XY Grid exploration is a method of exploring the parameter space of stable diffusion by generating a grid of images through varying two parameters. For example, steps and cfg scale can be varied to generate a grid of images with different values of these two parameters. This method can be useful for systematically exploring how changes in different parameters affect the output image. By generating a grid of images with different parameter values, it is possible to compare and analyze the effects of different parameter settings on the output image.

##### One Parameter
One parameter exploration involves generating a set of images by varying a single parameter, such as the delay in prompt delay. It can be useful for fine-tuning the impact of a particular parameter on image generation.




## Editing Composition
Tools in Stable Diffusion used to edit the composition of an image



### Image2Image
Img2img, or image-to-image, is a feature of Stable Diffusion that allows for image generation using both a prompt and an existing image. Users upload a base photo, and the AI applies changes based on entered prompts, resulting in refined and sophisticated art. The feature is similar to text-to-image generation, but with the added component of an existing image as a starting point. The possibilities for img2img generation are endless, with users experimenting with messy drawings, portraits, landscapes, and more to create a wide range of unique and creative artwork. The higher the denoising strength, the more different the image obtained will be.

#### Img2Img
If you like the general composition of the image but don't want to change very many of the details use img2img it with a lowish denoising strength. If you want to change it a lot more just use a higher denoising strength.

#### Inpainting
Inpainting is a feature of Stable Diffusion that allows users to change small details within an image composition. For example, if a user is creating scenery and wants to change part of a river, they can use inpainting to edit the river until it appears as desired. Similarly, if a user is creating a character and wants to add or edit features such as hands or a hat, they can use inpainting to make those changes. Inpainting uses specifically trained inpainting models that can be merged with other models. This feature enables users to create highly detailed and customized images with ease.

#### Outpainting
Outpainting is a feature in Stable Diffusion that allows you to extend the boundaries of your image to create a larger composition. For example, if you have a character that you want to show in a specific environment, you can use outpainting to gradually extend the scenery around the character to create a more complete and consistent image. The feature uses specifically trained models for outpainting and can be merged with other models for more creative possibilities.

#### Loopback
Loopback is a feature of Stable Diffusion where the output of image2image is fed into the input of the next image2image in a loop. This can be useful for creating a sequence of images with gradually decreasing changes between each image. By adjusting the denoising strength factor between each run, the number of changes can be progressively reduced, resulting in a smoother and more gradual transition between images. Loopback can also be used for creating animated sequences, where the output of each loop is fed into a video encoder to create a final animation.

#### InstructPix2Pix
InstructPix2Pix is a tool that allows users to provide natural language instructions to Stable Diffusion for changing specific parts of an image. It uses a Pix2Pix-based neural network to generate the changed image. Users can input a sentence such as "make the sky red" or "remove the trees," and the tool will generate a modified version of the original image according to the instruction. It provides an easy and intuitive way for users to edit their images without requiring specific technical knowledge or skills. The tool is available on GitHub for free use and experimentation.
https://github.com/timothybrooks/instruct-pix2pix 

#### Depth2Image
Depth2Image is a feature of Stable Diffusion that performs image generation similar to img2img, but also takes into account depth information estimated using the monocular depth estimator MIDAS. This allows for better preservation of composition in the generated image compared to img2img.

A depth-guided model, named "depth2img", was introduced with the release of Stable Diffusion 2.0 on November 24, 2022; this model infers the depth of the provided input image, and generates a new output image based on both the text prompt and the depth information, which allows the coherence and depth of the original input image to be maintained in the generated output.

https://zenn.dev/discus0434/articles/ef418a8b0b3dc0 (Japanese)

##### Depth Map
A depth map is an image that assigns a depth value to each pixel in a given image. It provides information about the distance of objects in the scene from the viewpoint of the camera. In the context of Stable Diffusion, a depth map can be used as a reference to generate images with higher accuracy and create 3D-like effects. It can also be used to separate objects and perform post-processing, such as creating videos. There are scripts available on GitHub that allow for depth map functionality in Stable Diffusion's web interface.
https://github.com/thygate/stable-diffusion-webui-depthmap-script

##### Depth Preserving Img2Img
Depth Preserving Image2Image is a feature in Stable Diffusion that preserves the depth information of the original image during the image generation process. This allows for more accurate and consistent results when applying prompts and generating new images. For example, if you want to cartoonize a photo, using a conventional Image2Image with a prompt may change the proportions and positioning of the elements in the image. However, with depth preserving Image2Image, the generated image will maintain the same proportions and positions as in the original photo, while still applying the desired style or effect. This allows for greater creative flexibility while preserving the composition of the original image.

#### ControlNet
ControlNet is an upgraded version of img2img that emphasizes edges and uses them in newly generated images. It refines the images by using special ControlNet models and can be used with any normal model. It allows for greater control of inputs and outputs and is ideal for coloring, filling in linework, texture reskin, style changes, or marking complex edges in an image that you don't want changed. ControlNet can also use scribbles as inputs and play well with larger and custom resolutions. The weights and models of ControlNet vary in their function and can include Midas/Depth, Canny-linework, HED-a mask, MLSD- for Architecture/Buildings/Straight Lines, OpenPose-pose transfer, and Scribble- a cross between Canny/HED for drawing scribbles. However, ControlNet has limitations with variation beyond "filling in" since it keeps the edges strongly. Overall, it is similar to Depth Maps, Normal Maps, and Holistically-nested edge detection. The ControlNet demo can be found on Hugging Face, and the research paper, repository, models, and tutorial can be found on GitHub.

Different models of this do different things, and weight of it affects it too
- Midas - Depth
- Canny - linework
- HED - a mask?
- MLSD - for Architecture / Buildings / Straight Lines
- OpenPose - can transfer a pose from one image to another
- Scribble - like a cross between Canny/HED but meant to be used for drawing scribbles

Demo - https://huggingface.co/spaces/hysts/ControlNet
Research Paper - https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/control.pdf 
Repo - https://github.com/lllyasviel/ControlNet 
Models - https://huggingface.co/lllyasviel/ControlNet
Compressed Models - https://huggingface.co/webui/ControlNet-modules-safetensors/tree/main
Automatic 1111 Addon - https://github.com/Mikubill/sd-webui-controlnet
Tutorial - https://youtu.be/vhqqmkTBMlU https://youtu.be/OxFcIv8Gq8o 



### Pix2Pix-zero
Pix2Pix-zero is an interactive image-to-image translation tool built on top of the Pix2Pix architecture. It allows users to sketch simple drawings, which are then transformed into a fully realized image by the model. The unique aspect of Pix2Pix-zero is that it is a zero-shot learning approach, meaning that it can generate images based on unseen or incomplete sketches.

The interface of Pix2Pix-zero is simple and easy to use, with a sketch pad on the left and a preview of the generated image on the right. Users can select from several different models trained on different datasets to generate images in different styles. The models are trained on datasets such as horses, shoes, and handbags.

The Pix2Pix-zero repository on GitHub includes pre-trained models as well as code for training your own models on custom datasets. Additionally, the website provides a live demo where users can try out the tool and generate their own images from sketches. Overall, Pix2Pix-zero provides an intuitive and interactive way for users to create images without needing advanced artistic skills.
https://pix2pixzero.github.io/
https://github.com/pix2pixzero/pix2pix-zero



### Seed Resize
Seed resize is a feature in Stable Diffusion that allows users to preserve the composition of an image while changing its size. Users can resize the seed image, which is the initial image that is fed into the image generation process to generate images of different sizes while maintaining the same composition. This feature is useful for creating images of different resolutions or aspect ratios without sacrificing the overall composition. It is also helpful in generating images for specific platforms or devices that require specific resolutions or sizes.

#### Variations
Variations are a feature of Stable Diffusion that allows for traversing latent space near the seed with a defined amount of difference. It generates a set of images that are similar to the original but with variations based on the given parameters. The variations can be used to explore different styles and variations for the same image, or to fine-tune the final output to the desired result. The feature can be useful in creating art that has a consistent theme or style while still being unique and interesting.




## Finishing
Finishing in Stable Diffusion refers to the final touches required to display the generated image. These include correcting any issues with faces using face restoration techniques. Once the image is satisfactory, it can be upscaled to the desired image size using SD upscaling, which is considered one of the best methods for this task. In some cases, inpainting can also be used to touch up small details after upscaling.


### Upscaling
Upscaling is a process of increasing the resolution of an image. In Stable Diffusion, images are usually generated at a lower resolution such as 512x512 or 768x768 for faster processing. However, to obtain higher-quality output or to use the generated image for printing or large displays, upscaling is necessary. There are various upscaling techniques available, including interpolation-based methods and deep learning-based methods. In Stable Diffusion, the preferred upscaling method is SD Upscale, which is a deep learning-based method specifically designed for stable diffusion.

https://upscale.wiki/wiki/Model_Database

#### BSRGAN
BSRGAN is a type of GAN (Generative Adversarial Network) that can be used for image super-resolution. It is designed to produce high-quality images with finer details and better textures than traditional methods. BSRGAN uses a combination of a generator network and a discriminator network to produce realistic images with high resolution. The generator network upscales a low-resolution image to a high-resolution image, while the discriminator network evaluates the quality of the generated image. The generator network is trained using a loss function that includes both adversarial loss and content loss. BSRGAN has been shown to produce high-quality super-resolved images in comparison to other state-of-the-art methods. The code for BSRGAN is available on GitHub.
https://github.com/cszn/BSRGAN

#### ESRGAN
ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) is an image upscaling method that uses deep neural networks to generate high-resolution images from low-resolution inputs. It was introduced in a 2018 research paper by Wang et al. and has since been widely used in image-processing tasks.

ESRGAN is based on the super-resolution GAN (SRGAN) model, which was introduced in 2017. However, ESRGAN improves upon SRGAN by incorporating residual blocks and a novel architecture called the "Enhancement Network" to enhance the high-frequency details in the output images. It also uses a perceptual loss function that takes into account both the content and style of the input image, resulting in more visually pleasing outputs.

ESRGAN has been used in a variety of applications, including image restoration, image super-resolution, and image synthesis. It has shown promising results in producing high-quality, detailed images from low-resolution inputs, making it a useful tool for various industries such as film, gaming, and art.

##### 4x RealESRGAN
4x RealESRGAN is an algorithm that is an upgrade to the ESRGAN algorithm. It is capable of upscaling images up to four times their original size while maintaining high image quality. RealESRGAN is based on deep neural networks and is trained on a large dataset of high-resolution images to learn how to upscale images without losing quality. The RealESRGAN algorithm can be accessed on GitHub, and a demo is available on the Hugging Face website.
https://github.com/xinntao/Real-ESRGAN
DEMO: https://huggingface.co/spaces/akhaliq/Real-ESRGAN

##### Lollypop
Lollipop is exceptional at making cartoon, manga, anime and pixel art content.

Lollypop upscaler is a universal model aimed at pre-rendered images, including realistic faces, manga, pixel art, and dithering. The model is trained using the patchgan discriminator with cx loss, cutmixup, and frequency separation, resulting in good results with a slight grain due to patchgan and sharpening using cutmixup. It can handle a variety of image types and is designed for upscaling images to a higher resolution.

##### Universal Upscaler
Seems well-liked, It comes with a different level of sharpness. Universal Upscaler Neutral, Universal Upscaler Sharp, Universal Upscaler Sharper. 

##### Ultrasharp
4x-ultrasharp is a powerful upscaling model that generates high amounts of detail and texture, particularly for images with JPEG compression. It can also restore highly compressed images. If a more balanced output is desired, the UltraMix Collection is recommended, which is a set of interpolated models based on UltraSharp and other models.

##### Uniscale
Uniscale is a tool that is useful for upscaling images, and it comes in various settings depending on whether the user wants a sharper or softer upscale. Some of these settings include Uniscale Balanced, Uniscale Strong, Uniscale V2 Soft, Uniscale V2 Moderate, Uniscale V2 Sharp, Uniscale NR Balanced, Uniscale NR Strong, and Uniscale Interp.

##### NMKD Superscale
NMKD Superscale is a model specifically designed for upscaling realistic images and photos that contain noise and compression artifacts. It is trained using a combination of adversarial and perceptual losses, which helps to preserve details and textures while removing artifacts. The model has been optimized for JPEG and WebP compressed images, making it well-suited for images downloaded from the internet or taken on a mobile device. NMKD Superscale has been well-received by users for its ability to produce high-quality upscaled images with minimal artifacts.

##### Remacri by Foolhardy
Remacri is an image upscaler that is an interpolated version of IRL models like Siax, Superscale, Superscale Artisoft, Pixel Perfect, and more. It is based on BSRGAN but has more details and less smoothing, which helps preserve features like skin texture and other fine details. The goal is to prevent images from becoming mushy and blurry during the upscaling process.

#### SD Upscale
SD Upscale is a method of upscaling images that uses Stable Diffusion to add details tile by tile after upscaling with a conventional upscaler. This is done to avoid running out of VRAM when processing the entire upscaled image. Any Stable Diffusion checkpoint can be used for this process. For example, an image can be generated using Stable Diffusion 1.5 and then upscaled using the depth model, or it can be generated using Stable Diffusion 2.1 and then upscaled using Robodiffusion.

##### SD 2.0 4xUpscaler
SD 2.0 4x Upscaler is the official model from stability.ai that allows for upscaling images by a factor of four. However, it requires a lot of VRAM to use, which can be a limitation for some users.


### Restoring
Restoring is a process of fixing and improving the quality of an image. It can involve sharpening the image to enhance its details, or it can be used to fix specific issues like smoothing out skin textures or removing noise and artifacts. Restoring can be performed using various techniques and algorithms, depending on the specific needs of the image. For example, face restoration can be used to improve the quality of facial features and expressions, while denoising algorithms can be used to remove unwanted noise and improve the clarity of the image. Restoring is an important step in the image creation process to ensure that the final product is of high quality and meets the desired standards.

#### Face Restoration
Face restoration algorithms are used to adjust the details of a face in an image, such as the eyes, skin texture, and overall clarity. These algorithms use machine learning techniques to identify facial features and make targeted adjustments to improve the overall appearance of the face. They can be used to enhance the quality of portrait photographs, as well as to correct facial imperfections or blemishes. Some popular face restoration algorithms include DeepFaceLab, Faceswap, and OpenCV.

##### GFPGAN
GFPGAN is an algorithm that uses StyleGAN for face restoration. The algorithm is based on a generative adversarial network that is trained to generate high-quality images of faces. It can be used for tasks such as face super-resolution, face inpainting, and face colorization. GFPGAN is an improvement over previous face restoration algorithms because it is able to produce more realistic results with better detail and texture. It is open source and available on GitHub, and a demo can be found on Hugging Face.
https://github.com/TencentARC/GFPGAN
DEMO: https://huggingface.co/spaces/akhaliq/GFPGAN

##### Code Former
Code Former is a face restoration algorithm that utilizes a convolutional neural network (CNN) to restore and refine facial features. The algorithm uses an encoder-decoder architecture with skip connections to effectively capture facial features and details while maintaining a smooth output. It also incorporates adversarial training to improve the realism of the output. The Code Former algorithm can be implemented using Python and Tensorflow. It has been shown to produce high-quality results in facial restoration tasks.
https://github.com/sczhou/CodeFormer
DEMO: https://huggingface.co/spaces/sczhou/CodeFormer



## Software Addons  

### Blender Addons  
#### Blender ControlNet
- https://github.com/coolzilj/Blender-ControlNet
#### Makes Textures / Vision
- https://www.reddit.com/r/blender/comments/11pudeo/create_a_360_nonerepetitive_textures_with_stable/
#### OpenPose
- https://gitlab.com/sat-mtl/metalab/blender-addon-openpose
#### OpenPose Editor
- https://github.com/fkunn1326/openpose-editor
#### Dream Textures
- https://github.com/carson-katri/dream-textures https://www.youtube.com/watch?v=yqQvMnJFtfE https://www.youtube.com/watch?v=4C_3HCKn10A, similar to materialize https://boundingboxsoftware.com/materialize/ https://github.com/BoundingBoxSoftware/Materialize
#### AI Render
- https://blendermarket.com/products/ai-render https://www.youtube.com/watch?v=goRvGFs1sdc https://github.com/benrugg/AI-Render https://airender.gumroad.com/l/ai-render https://blendermarket.com/products/ai-render https://www.youtube.com/watch?v=tmyln5bwnO8 https://github.com/benrugg/AI-Render/wiki/Animation
#### Stability AI's official Blender
- https://platform.stability.ai/docs/integrations/blender
#### CEB Stable Diffusion (Paid)
- https://carlosedubarreto.gumroad.com/l/ceb_sd  
#### Cozy Auto Texture
- https://github.com/torrinworx/Cozy-Auto-Texture

### Blender Rigs/Bones  
#### ImpactFrames' OpenPose Rig
- https://ko-fi.com/s/f3da7bd683 https://impactframes.gumroad.com/l/fxnyez https://www.youtube.com/watch?v=MGjdLiz2YLk https://www.reddit.com/r/StableDiffusion/comments/11cxy5h/comment/jacorrt/?utm_source=share&utm_medium=web2x&context=3
#### ToyXYZ's Character bones that look like Openpose for blender
- https://toyxyz.gumroad.com/l/ciojz script to help it https://www.reddit.com/r/StableDiffusion/comments/11fyd6q/blender_script_for_toyxyzs_46_handfootpose/
#### 3D posable Mannequin Doll
- https://www.artstation.com/marketplace/p/VOAyv/stable-diffusion-3d-posable-manekin-doll https://www.youtube.com/watch?v=MClbPwu-75o
#### Riggify model
- https://3dcinetv.gumroad.com/l/osezw  
- 

### Maya
#### ControlNet Maya Rig
- https://impactframes.gumroad.com/l/gtefj https://youtu.be/CFrAEp-qSsU  

### Photoshop  
#### Stable.Art
- https://github.com/isekaidev/stable.art
#### Auto Photoshop Plugin
- https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin  

### Daz
#### Daz Control Rig
- https://civitai.com/models/13478/dazstudiog8openposerig

### Cinema4D
#### Colors Scene (possibly no longer needed since controlNet Update)
- https://www.reddit.com/r/StableDiffusion/comments/11flemo/color150_segmentation_colors_for_cinema4d_and/



## Related Technologies, Communities and Tools, not necessarily Stable Diffusion, but Adjacent
DeepDream
- https://deepdreamgenerator.com/

StylGAN Transfer

AI Colorizers  
- DeOldify

- Style2Paint https://github.com/lllyasviel/style2paints

## Techniques & Possibilities
### Clip Skip & Alternating
CLIP-Skip is a slider option in the settings of Stable Diffusion that controls how early the processing of prompt by the CLIP network should be stopped. It is important to note that CLIP-Skip should only be used with models that were trained with this kind of tweak, which in this case are the NovelAI models. When using CLIP-Skip, the output of the neural network will be based on fewer layers of processing, resulting in better image generation on the appropriate models.
https://www.youtube.com/watch?v=IkMIoRCfCgE
https://www.reddit.com/r/StableDiffusion/comments/yj58r0/psa_clipskip_should_only_be_used_with_models/

### Multi Control Net and blender for perfect Hands
https://www.youtube.com/watch?v=ptEZQrKgHAg&t=4s

### Blender to Depth Map
https://www.reddit.com/r/StableDiffusion/comments/115ieay/how_do_i_feed_normal_map_created_in_blender/

Many use freestyle to controlNet instead, claim it gives best results

https://www.reddit.com/r/StableDiffusion/comments/zh8ava/comment/izks993/?utm_source=share&utm_medium=web2x&context=3
https://stable-diffusion-art.com/depth-to-image/

#### Blender to depth map for concept art
https://www.youtube.com/watch?v=L6J4IGjjr9w

#### depth map for terrain and map generation?


### Blender as Camera Rig
https://www.reddit.com/r/StableDiffusion/comments/10fqg7u/quick_test_of_ai_and_blender_with_camera/


### SD depthmap to blender for stretched single viewpoint depth perception model
https://www.youtube.com/watch?v=vfu5yzs_2EU https://github.com/Ladypoly/Serpens-Bledner-Addons importdepthmap  

similar to https://huggingface.co/spaces/mattiagatti/image2mesh https://towardsdatascience.com/generate-a-3d-mesh-from-an-image-with-python-12210c73e5cc  
similar to https://github.com/hesom/depth_to_mesh

### Daz3D for posing
https://www.reddit.com/r/StableDiffusion/comments/11owo31/comment/jbvdmsm/?utm_source=share&utm_medium=web2x&context=3

### Mixamo for Posing
https://www.reddit.com/r/StableDiffusion/comments/11owo31/something_that_might_help_ppl_with_posing/

### Figure Drawing Poses as Reference Poses
https://figurosity.com/figure-drawing-poses


### Generating Images to turn into 3D sculpting brushes
https://www.reddit.com/r/StableDiffusion/comments/xjju0q/ai_generated_3d_sculpting_brushes/


### Stable Diffusion to Blender to create particles using automesh plugin
https://twitter.com/subcivic/status/1570754141995290626  
https://wesxdz.gumroad.com/l/xfdmzx  

## Not Stable Diffusion But Relevant Techniques
3D photo effect https://shihmengli.github.io/3D-Photo-Inpainting/

## Other Resources

### API's

NextML API for STable Diffusion https://api.stable-diffusion.nextml.com/redoc

DreamStudio API