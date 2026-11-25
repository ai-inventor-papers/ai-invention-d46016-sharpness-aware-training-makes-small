# Style exemplars

Field: calibration and optimisation of deep classifiers (machine learning, image classification). Papers: Tan et al. (ICLR 2026, arXiv 2505.23866), Guo et al. (ICML 2017, arXiv 1706.04599), Nixon et al. (CVPR Workshops 2019, arXiv 1904.01685), Foret et al. (ICLR 2021, arXiv 2010.01412), Morosini et al. (2026, arXiv 2604.20614).

Register: sentences run long and carry several clauses, first person plural "we" is used for the authors' own findings, hedges are few and placed where the evidence is thin, and citations are dense inside the sentences that rest on them.

## Passage 1: Abstract, Tan et al. (ICLR 2026)
Source: https://arxiv.org/abs/2505.23866

Deep neural networks have been increasingly used in safety-critical applications
such as medical diagnosis and autonomous driving. However, many studies suggest
that they are prone to being poorly calibrated and have a propensity for overcon-
fidence, which may have disastrous consequences. In this paper, unlike standard
training such as stochastic gradient descent, we show that the recently proposed
sharpness-aware minimization (SAM) counteracts this tendency towards overconfidence. The theoretical analysis suggests that SAM allows us to learn models that
are already well-calibrated by implicitly maximizing the entropy of the predictive
distribution. Inspired by this finding, we further propose a variant of SAM, coined
as CSAM, to ameliorate model calibration. Extensive experiments on various
datasets, including ImageNet-1K, demonstrate the benefits of SAM in reducing calibration error. Meanwhile, CSAM performs even better than SAM and consistently
achieves lower calibration error than other approaches.

## Passage 2: Abstract, Guo et al. (ICML 2017)
Source: https://arxiv.org/abs/1706.04599

Confidence calibration – the problem of predicting probability estimates representative of the true correctness likelihood – is important for classification models in many applications. We discover that modern neural networks, unlike those from a decade ago, are poorly calibrated. Through extensive experiments, we observe that depth, width, weight decay, and Batch Normalization are important factors influencing calibration. We evaluate the performance of various post-processing calibration methods on state-of-the-art architectures with image and document classification datasets. Our analysis and experiments not only offer insights into neural network learning, but also provide a simple and straightforward recipe for practical settings: on most datasets, temperature scaling – a single-parameter variant of Platt Scaling – is surprisingly effective at calibrating predictions.

## Passage 3: Introduction, Guo et al. (first paragraph)
Source: https://arxiv.org/abs/1706.04599

Recent advances in deep learning have dramatically improved neural network accuracy (Simonyan & Zisserman, 2015; Srivastava et al., 2015; He et al., 2016; Huang et al., 2016; 2017). As a result, neural networks are now entrusted with making complex decisions in applications, such as object detection (Girshick, 2015), speech recognition (Hannun et al., 2014), and medical diagnosis (Caruana et al., 2015). In these settings, neural networks are an essential component of larger decision making pipelines.

## Passage 4: Argument passage, Nixon et al. (CVPR Workshops 2019)
Source: https://arxiv.org/abs/1904.01685

The reason that we need to assess the quality of calibrated probabilities is that presently, we optimize against a proper scoring rule and our models often yield uncalibrated probabilities. Recent work proposed Expected Calibration Error (ECE; Naeini et al., 2015), a measure of calibration error which has lead to a surge of works developing methods for calibrated deep neural networks (e.g., Guo et al., 2017; Kuleshov et al., 2018). In this paper, we show that ECE has numerous pathologies, and that recent calibration methods, which have been shown to successfully recalibrate models according to ECE, cannot be properly evaluated via ECE.

## Section outlines
- Tan et al. (ICLR 2026): Abstract; Introduction; Related Work; Preliminaries (Expected Calibration Error); Methodology; Theoretical Analysis; Improving SAM Towards Better Calibration; Experiments; CSAM Even Calibrates Better Than SAM; Conclusion; Acknowledgments; References.
  - Method is organised by component: the theoretical analysis, then the proposed variant CSAM.
  - Results are organised by research question and dataset: the main comparison, then a dedicated section on the variant against SAM.
- Guo et al. (ICML 2017): Abstract; 1. Introduction; further sections were not recoverable from the saved full text.
  - Method is organised by post-processing method (temperature scaling and the other calibrators).
  - Results are organised by architecture and dataset, with ablations on depth, width, weight decay and batch normalisation.
- Nixon et al. (CVPR Workshops 2019): Abstract; further section headings were not recoverable from the saved full text.
- Foret et al. (ICLR 2021) and Morosini et al. (2026): full text not structured in the saved copy; headings not extracted.

## Gaps
- No results passage with numbers was copied from Guo or Nixon; the discussion passage is not yet extracted. The register notes above rest on the four passages copied.
