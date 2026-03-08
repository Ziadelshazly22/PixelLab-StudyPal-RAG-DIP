# Evaluation Report — DIP AI Tutor

_Generated: 2026-03-08 04:16 UTC_

## Summary

| Overall Score | Questions | Metrics Passing | Status |
| --- | --- | --- | --- |
| **0.790 / 1.000** | 15 DIP + 3 guardrail | 4 / 4 | ✅ All metrics pass |

## RAGAS Scores

| Metric | Score | Target | Status |
| --- | --- | --- | --- |
| faithfulness | 0.726 | 0.700 | ✅ PASS |
| answer_relevancy | 0.807 | 0.700 | ✅ PASS |
| context_precision | 0.918 | 0.700 | ✅ PASS |
| context_recall | 0.709 | 0.700 | ✅ PASS |
| **Overall (mean)** | **0.790** | 0.700 | ✅ PASS |

## Per-Topic Breakdown

| Topic | Faithfulness | Answer Relevancy |
| --- | --- | --- |
| 2D Discrete Fourier Transform — definition and properties | 1.000 | 0.689 |
| Canny edge detector — five steps | 0.045 | 0.882 |
| Frequency domain filtering — ideal vs Gaussian low-pass | 0.000 | 0.658 |
| Gaussian noise model — probability density function | 0.750 | 0.888 |
| Geometric transformation — bilinear interpolation for image resampling | 1.000 | 0.803 |
| Histogram equalization — derivation of the transformation function | 0.857 | 0.977 |
| Histogram specification / matching | 0.500 | 0.563 |
| Image segmentation — thresholding (Otsu's method) | 0.846 | 0.976 |
| JPEG compression — DCT and quantization | 0.733 | 0.837 |
| Morphological dilation — dual relationship with erosion | 0.786 | 0.794 |
| Morphological erosion — definition and structuring element | 0.929 | 0.778 |
| Opening and closing operations — applications | 0.615 | 0.857 |
| Sobel edge detection — gradient masks | 1.000 | 0.889 |
| Spatial filtering — linear filters, convolution | 1.000 | 0.786 |
| Spatial filtering — nonlinear (median filter, salt-and-pepper noise) | 0.833 | 0.735 |

## Latency Analysis

- **Mean**: 23.78 s
- **p50**: 22.05 s
- **p95**: 29.36 s ⚠️ exceeds 5 s SLA target

## Guardrail Test Results

| Question | Status | Answer Preview |
| --- | --- | --- |
| What is the boiling point of water? | ✅ PASS | This question falls out of focus... |
| Explain the rules of chess. | ✅ PASS | This question falls out of focus... |
| How do I make pasta bechamel in the best way? | ✅ PASS | This question falls out of focus... |

## Individual Questions Below Target

**Q3** — What is the 2D Discrete Fourier Transform and what are its key properties?

- `answer_relevancy`: 0.689
- `context_precision`: 0.700

**Q4** — What is the difference between an ideal low-pass filter and a Gaussian low-pass filter?

- `faithfulness`: 0.000
- `answer_relevancy`: 0.658

**Q6** — What is histogram specification and how does it differ from histogram equalization?

- `faithfulness`: 0.500
- `answer_relevancy`: 0.563

**Q7** — How is morphological erosion defined and what role does the structuring element play?

- `context_recall`: 0.000

**Q9** — What are morphological opening and closing operations and what are they used for?

- `faithfulness`: 0.615

**Q11** — What are the five steps of the Canny edge detector?

- `faithfulness`: 0.045

**Q12** — How does Otsu's method determine the optimal threshold for image segmentation?

- `context_precision`: 0.625

**Q15** — How is bilinear interpolation used for geometric image resampling?

- `context_recall`: 0.211
