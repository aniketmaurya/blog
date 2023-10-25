---
aliases:
- /Object Detection/2020/01/13/EfficientDet
badges: true
categories:
- Object Detection
date: '2020-01-13'
description: EfficientDet, highly efficient and scalable state of the art object detection
  model developed by Google Research, Brain Team.
image: https://raw.githubusercontent.com/aniketmaurya/machine_learning/master/blog_files/2020-01-13-EfficientDet/compare.png
keywords: object detection, efficientdet, google, automl, efficientnet
layout: post
title: 'EfficientDet: When Object Detection Meets Scalability and Efficiency'
toc: true

---

EfficientDet, a highly efficient and scalable state of the art object detection model developed by Google Research, Brain Team. It is not just a single model. It has a family of detectors which achieve better accuracy with an order-of-magnitude fewer parameters and FLOPS than previous object detectors.

EfficientDet paper has mentioned its 7 family members.

> ### Comparison of EfficientDet detectors[0–6] with other SOTA object detection models.


![Source: [arXiv:1911.09070v1](https://arxiv.org/abs/1911.09070)](https://cdn-images-1.medium.com/max/3360/1*wVXzRV58CNjHMV24FMFz7g.png)*Source: [arXiv:1911.09070v1](https://arxiv.org/abs/1911.09070)*

## Quick Overview of the Paper

1. **[EfficientNet](https://arxiv.org/abs/1905.11946)** is the backbone architecture used in the model. EfficientNet is also written by the same authors at Google. Conventional CNN models arbitrarily scaled network dimensions- *width, depth and resolution*. EfficientNet uniformly scales each dimension with a fixed set of scaling coefficients. It surpassed SOTA accuracy with 10x efficiency.

1. **BiFPN**: While *fusing* (applying [residual or skip connections](https://arxiv.org/abs/1512.03385)) different input features, most of the works simply summed them up without any distinction. Since both input features are at the different resolutions they don’t equally contribute to the fused output layer. The paper proposes a weighted bi-directional feature pyramid network (BiFPN), which introduces learnable weights to learn the importance of different input features.

1. **Compound Scaling**: For higher accuracy previous object detection models relied on — bigger backbone or larger input image sizes. Compound Scaling is a method that uses a simple compound coefficient **φ** to jointly scale-up all dimensions of the backbone network, BiFPN network, class/box network, and resolution.

> Combining EfficientNet backbones with our propose BiFPN and compound scaling, we have developed a new family of object detectors, named EfficientDet, which consistently achieve better accuracy with an order-of-magnitude fewer parameters and FLOPS than previous object detectors.

## BiFPN

![Source: [arXiv:1911.09070v1](https://arxiv.org/abs/1911.09070) — figure 2](https://cdn-images-1.medium.com/max/2000/1*ZqkRP7n8Xh-PILCuX1510A.png)*Source: [arXiv:1911.09070v1](https://arxiv.org/abs/1911.09070) — figure 2*

Conventional FPN (Feature Pyramid Network) is limited by the one-way information flow. [PANet](https://arxiv.org/pdf/1803.01534.pdf) added an extra bottom-up path for information flow. PANet achieved better accuracy but with the cost and more parameters and computations. The paper proposed several optimizations for cross-scale connections:

1. Remove Nodes that only have one input edge.
*If a node has only one input edge with no feature fusion, then it will have less contribution to the feature network that aims at fusing different features.*

2. Add an extra edge from the original input to output node if they are at the same level, in order to fuse more features without adding much cost.

3. Treat each bidirectional (top-down & bottom-up) path as one feature network layer, and repeat the same layer multiple times to enable more high-level feature fusion.

## Weighted Feature Fusion

While multi-scale fusion, input features are not simply summed up. The authors proposed to add additional weight for each input during feature fusion and let the network to learn the importance of each input feature. Out of three weighted fusion approaches —
**Unbounded fusion:**

![Source: [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)](https://cdn-images-1.medium.com/max/2000/1*rv-MpuUiv-uVs5Trx4XI1Q.png)*Source: [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)*

Where *W* is a learnable weight that can be a scalar (per-feature), a vector (per-channel), or a multi-dimensional tensor (per-pixel). Since the scalar weight is unbounded, it could potentially cause training instability. So, Softmax-based fusion was tried for normalized weights.
**Softmax-based fusion:**

![Source: [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)](https://cdn-images-1.medium.com/max/2000/1*AhA54rA3NcNFEuosOac_YA.png)

As softmax normalizes the weights to be the probability of range 0 to 1 which can denote the importance of each input. The softmax leads to a slowdown on GPU.
**Fast normalized fusion:**

![Source: [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)](https://cdn-images-1.medium.com/max/2000/1*AO0A5dDSJl0GKeUvYYc4cQ.png)

Є is added for numeric stability. It is 30% faster on GPU and gave almost as accurate results as softmax.
> Final BiFPN integrates both the bidirectional cross-scale connections and the fast normalized fusion.

## EfficientDet Architecture

![Source: [arXiv:1911.09070v1](https://arxiv.org/abs/1911.09070) — figure 3](https://cdn-images-1.medium.com/max/4062/1*nP0LlBoz0Uqhd17T4bINzg.png)*Source: [arXiv:1911.09070v1](https://arxiv.org/abs/1911.09070) — figure 3*

EfficientDet follows one-stage-detection paradigm. A pre-trained EfficientNet backbone is used with BiFPN as the feature extractor. BiFPNN takes {P3, P4, P5, P6, P7} features from the EfficientNet backbone network and repeatedly applies bidirectional feature fusion.
The fused features are fed to a class and bounding box network for predicting object class and bounding box.


## References

[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

[Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)

[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

<hr>
<br>
> Hope you liked the article.

👉 [Twitter](https://twitter.com/aniketmaurya): [https://twitter.com/aniketmaurya](https://twitter.com/aniketmaurya)
👉 Mail: aniketmaurya@outlook.com
