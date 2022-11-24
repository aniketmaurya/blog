---
title: 'Stable Diffusion 2.0'
description: 'What is new in Stable Diffusion 2.0'
badges: true
categories:
- text-to-image
- image generation
image: sd2release.png
date: '2022-11-24'
keywords: image generation, stable diffusion, deep learning,
layout: post
toc: true
---

While the world was already amazed by the performance of open source text to image model, Stable Diffusion 1.x. Stability AI has released a new version with a lot of improvements.

![](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/assets/stable-samples/txt2img/768/merged-0006.png)
![](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/assets/stable-samples/txt2img/merged-0003.png)


## List of notable updates:

* Trained using a new text encoder, OpenCLIP, developed by LAION with support from Stability AI.
* The text-to-image models can now generate images with default resolutions of both 512x512 pixels and 768x768 pixels.
* The models are trained on subset of LAION-5B dataset after filtering out adult content using NSFW filter.
* Stable Diffusion 2.0 comes with an Upscaler Diffusion model that enhances the resolution of images by a factor of 4.
* Depth-to-Image: It extends the version 1 image-to-image by also considering depth of an input image for generation of a new image.
* It also brings a new text guided image inpainting diffusion model, finetuned on the new Stable Diffusion 2.0 base text-to-image.
