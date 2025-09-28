# From-Baselines-to-Robust-Pipelines-Spatial-Relation-Reasoning-in-CT
Hybrid pipeline for spatial relation reasoning in CT imaging, combining LLM-based query parsing with medical segmentation and deterministic geometry checks.  

Understanding spatial relations between anatomical structures (e.g., “Is the left kidney below the inferior vena cava?”) is a key building block for automated radiological reporting. While vision-language models (VLMs) show strong performance in natural language understanding, they remain unreliable when precise geometric reasoning is required.  

This repository explores the progression from simplified baseline approaches to a hybrid pipeline that separates linguistic parsing from geometric computation:  

- Baselines  
  - PNG snapshots: convenient for visualization, but they lack Hounsfield Units, voxel spacing, orientation, and 3D consistency.
  - Vision-Language Models only: effective at handling synonyms and query parsing, but inaccurate in spatial relations due to missing coordinate grounding.
  - Prompt-engineering / Chain of Thought tricks: may improve language outputs but cannot solve fundamental geometric inconsistencies. May even lead to more hallucinations. The classification probably doesn't get much better. A few percent maybe, but more complex            workflow.
  - Skipping preprocessing: leads to orientation errors, distorted distances, and poor generalization across scanners and protocols.
 
You could use PNG snapshots and skip preprocessing if you only want to do a proof-of-concept with 2D-Slices.  
Problem: Detector needs to be trained again on ct scans if a full radiological examination is required. For a final, research-grade workflow → you need a full 3D detector/segmenter on DICOM/NIfTI.  
Training a PNG model is only worth it for short-term prototypes.  

## Why Radiological Analysis Cannot Be Done on PNG Images

Working with PNG exports instead of DICOM/NIfTI volumes is fundamentally limiting. PNGs are suitable for demonstrations and baselines, but **not** for radiological analysis. Three key reasons:

1. **Loss of Hounsfield Units (HU)**  
   - CT intensity values are mapped to windowed grayscale in PNGs.  
   - Absolute tissue densities (e.g., fat ≈ –100 HU, bone > +700 HU) are lost.  
   - Without HU, no meaningful tissue characterization, lesion assessment, or quantitative radiomics can be performed.  

2. **No Voxel Spacing or Patient Coordinates**  
   - PNGs lack information on voxel size (mm per pixel) and slice position/orientation.  
   - Distances, volumes, and anatomical relations in 3D cannot be measured.  
   - “Left/right” and “above/below” in the image may not correspond to patient anatomy.  

3. **Non-reproducible exports**  
   - PNGs depend on windowing, cropping, and orientation chosen during export.  
   - Two exports of the same CT series can look different, with no standardization.  
   - This breaks reproducibility and makes clinical or research validation impossible.  

---

✅ **Conclusion:**  
PNGs are acceptable for **baselines and prototyping** (e.g., testing multimodal LLMs), but radiological investigations require **DICOM or NIfTI** with full metadata.


## With full radiological examination that can be used in real world scenarios:  
Other data is needed (DICOM/NIfTI files), CT-scan normalization (for example isotropic resampling). And then also specific detectors and not only multimodal models like Pixtral-12B.


## With more simplified model:  
**Towards robust pipelines:**    
  - Language parsing (LLM-based): normalize synonyms and relations, map terms to ontology classes.
  - Medical segmentation/detection: extract centroids or masks from CT volumes in a normalized patient coordinate system.
  - Deterministic geometry checker: compute relations (left/right, above/below) using coordinates rather than heuristics.
  - QA and fallbacks: confidence filtering, anatomical plausibility checks, consistency across slices.

By grounding relation reasoning in image-space coordinates rather than purely text-based heuristics, this approach achieves:  
- Robustness within a fixed image orientation: If PNGs are exported consistently (same plane, same orientation), relations such as left/right or above/below can be determined reliably in image space.  
- Scalability to many queried relations: Once a detector provides pixel coordinates for multiple objects, arbitrary spatial relations can be computed deterministically.  
- Explainability through explicit pixel coordinates and distances: Results can be traced back to centroid positions or bounding boxes in the image (e.g., centroid A at x=120 vs. centroid B at x=300 → A is left of B).  
- A reproducible foundation for baseline experiments: While not clinically valid, a PNG-based setup allows reproducible baselines and prototypes to evaluate the strengths and weaknesses of vision-language models versus hybrid approaches.


## Pipeline for Relation Reasoning

### 1. Input
- PNG slice (baseline setup) + free-text question

### 2. Query Parsing (LLM)
- Task: normalize synonyms, handle plural/sides, and extract relation
- Output: structured schema for class lookup
```json
{ "obj1": "left kidney", "cls1": 3, "obj2": "ivc", "cls2": 63, "relation": "below" }
```

Parsing is required before detection, because the detector only knows fixed class IDs.
Without this step, synonyms like “vena cava inferior” or “IVC” cannot be mapped consistently.  

### 3. Detection / Segmentation
- **Model:** 2D detector or segmenter (e.g., YOLO, U-Net) on PNG slices  
- **Output:** pixel coordinates of centroids with confidence

```json
{ "cls": 3, "coords": [x,y], "score": 0.92 }
```


### 4. Geometry Checker (Image-Space)
- Rule-based comparison of centroids:
  - left/right → compare x
  - above/below → compare y
- Output: decision + justification
  
```json
{
  "answer": true,
  "reason": "Left kidney centroid (x=120,y=200) lies below IVC centroid (x=118,y=80).",
  "confidence": 0.87
}

```
The output is for example the structured JSON returned by the Geometry Checker.

### 5. LLM Explanation

- **Natural language explanation:**
Convert the structured JSON into a human-readable statement.  
Example: "Yes, the left kidney is located below the inferior vena cava in this slice, with high confidence (87%)."  
- **Contextualization:**
Add anatomical context or synonyms to improve readability for medical users.
- **Fallback handling:**
If confidence is too low, generate a cautious answer.
Example: "The relation could not be determined with sufficient confidence. Likely candidates are..."

## Small Alternative: LLM-generated coordinates + Geometry Checker

Instead of asking the LLM to directly answer *true/false*,  
we prompt it to output **estimated coordinates** of the queried objects in JSON form:

```json
{
  "obj1": "left kidney",
  "coords1": [120, 200],
  "obj2": "ivc",
  "coords2": [118, 80]
}
```

A deterministic Geometry Checker then evaluates the relation:

- left/right → compare x
- above/below → compare y

**Pros:**  
- Transparent and explainable
- Errors can be attributed to wrong coordinates (not the decision rule)
- Allows systematic evaluation (LLM parsing vs. geometric rule)
- Could have 10-20 % better accuracy

**Cons:**  
- Less reliable than a detector/segmenter trained for geometry

This setup is a stronger baseline than pure LLM yes/no predictions, but it cannot replace a proper detector-based pipeline.

## Problems with current Pixtral-12B:

The model is trained to predict the next text token on interleaved image and text data.  
This means the vision transformer is trained on token loss. This cannot be used for reliable and exact geometric reasoning.    


## Why Chain-of-Thought (CoT) is not effective for relational comparisons on PNGs

1) **No grounding in pixel coordinates**  
   - LLMs (even multimodal) do not output measured coordinates.  
   - CoT just verbalizes "steps" in language, but these steps are not tied to actual pixel positions.  
   - Example: The model may reason *“the kidney is usually lower than the liver”* → this is prior knowledge, not an observation of the given PNG.

2) **Language-based heuristics instead of visual evidence**  
   - CoT relies on text-based reasoning patterns.  
   - Relations in images (left/right, above/below) require **quantitative comparison** of pixel values, not narrative heuristics.  
   - The chain of reasoning does not change the underlying fact: the model is still guessing based on associations.

3) **Stochastic and non-reproducible outputs**  
   - CoT increases output length and reasoning variance.  
   - The same PNG can yield different relational answers depending on prompt phrasing or random sampling.  
   - For binary spatial relations, this randomness directly hurts accuracy.

4) **No improvement in accuracy for geometry**  
   - Empirically, CoT improves tasks that need multi-step **logical deduction** (math, text puzzles).  
   - Geometric relations are **atomic comparisons** (x₁ < x₂).  
   - Adding intermediate reasoning steps does not make the underlying guess any closer to the true pixel geometry.

5) **False explainability**  
   - CoT outputs *sound plausible* (“the object appears below because kidneys are lower”), but they are not tied to actual visual evidence.  
   - This creates an illusion of reasoning without measurable grounding in the PNG image.

---

**Conclusion:**  
On PNG-only setups, CoT cannot improve relational accuracy.  
It produces longer, more plausible-sounding answers, but still lacks grounding in pixel coordinates.  
For spatial relations, deterministic rules on extracted centroids (via a detector/segmenter) are the only reliable way forward.

In radiological examinations we can't guess, we need reliable systems that can be explained and work well.  



## Example Thesis Structure 

1. **Introduction & Related Work**  
   - Motivation: automated relation reasoning in medical imaging  
   - Trends: multimodal LLMs, Chain-of-Thought prompting  

2. **Baselines**  
   - **LLM direct output (yes/no)**  
   - **LLM + Chain-of-Thought (parsing, relational reasoning)**  
   - Hypothesis: CoT helps in text parsing but not in image geometry  

3. **Proposed Hybrid Pipeline**  
   - LLM for query parsing (normalize objects, extract relation)  
   - Detector/segmenter to provide coordinates  
   - Deterministic geometry checker for relations  

4. **Experiments & Evaluation**  
   - Compare accuracy: direct LLM vs. LLM+CoT vs. Hybrid  
   - Error analysis: parsing vs. geometry failures  
   - Discussion: explain why CoT fails, why hybrid is robust  

5. **Conclusion**  
   - CoT is useful for parsing but not for geometric reasoning  
   - Hybrid pipeline provides explainability, robustness, and scalability


## Why marked images yielded better results with Prompting

In our experiments, multimodal LLMs performed almost at chance level (~50%) when asked to solve spatial relation tasks directly on unmarked PNGs.  
When the same images contained **markings on the target organs**, performance improved by almost 10% for some datasets.  
This effect can be explained as follows:

1. **Localization is trivialized**  
   - Without markings, the LLM must both *detect the organ* and *compare positions*.  
   - With markings, the organ is visually highlighted → the localization step is solved for the model.
   - Prompts that focus on visual relations focus more on "visual tokens" that can be ignored otherwise  

2. **Strong saliency cues**  
   - LLM vision encoders respond much more reliably to high-contrast overlays (colored boxes, arrows) than to subtle grayscale textures in CT images.  
   - The model can use these cues to anchor its reasoning.  

3. **Reduced ambiguity, simpler reasoning**  
   - The LLM no longer has to decide *which structure is which*.  
   - It only needs to compare the relative positions of the highlighted regions.

**Limitation:**  
- Even with markings, the improvement remains modest (~10%).  
- The model still reasons via **visual heuristics**, not via deterministic geometry.  
- Markings remove the hardest problem (object localization), but they do not provide true **patient-space coordinates**.
- Chain-of-Thought prompting can add a **small further gain** (a few percentage points),  
  since it forces the LLM to verbalize object and relation steps.  
  However, this does not overcome the fundamental lack of geometric grounding.  


**Conclusion:**  
Markings make PNG-based CoT/prompting setups appear stronger by simplifying localization.  
However, this does not scale to unmarked datasets or clinical tasks. For reproducible relation reasoning, explicit detection/segmentation and coordinate-based geometry checks are required.  


## Why CoT yields only a few extra percentage points — and not more

In marked-image setups, we could observe that Chain-of-Thought (CoT) prompting can add a **small further gain** (a few percentage points) on top of the ~10% improvement from markings.  
This limited effect can be explained by two opposing factors:

### Why CoT can help a little
1. **Explicit step-by-step parsing**  
   - CoT forces the model to write down intermediate steps (*“object A is here → object B is here → compare positions”*).  
   - This reduces random yes/no guessing and activates more structured use of the visual input.  

2. **Activation of prior knowledge**  
   - LLMs carry statistical knowledge about anatomy (e.g., “kidneys are usually below the liver”).  
   - CoT gives the model space to incorporate such priors explicitly into its reasoning process.  

3. **Bias reduction through verbalization**  
   - Forcing reasoning chains can reduce shortcut behavior (e.g., always answering “left”).  
   - This leads to slightly more balanced and accurate predictions.  

### Why CoT cannot help much more
1. **Relations are atomic comparisons**  
   - Left/right or above/below are simple `x₁ < x₂` decisions.  
   - They are not multi-step logical puzzles where CoT typically shines.  

2. **No geometric grounding**  
   - The model does not measure pixel coordinates; it only generates plausible narratives.  
   - CoT cannot turn ungrounded heuristics into deterministic geometry.  

3. **Variance and instability**  
   - Longer reasoning chains introduce more stochasticity.  
   - Gains are small and inconsistent across datasets and prompts.
   - Less random errors by introducing step by step reasoning: Can also be obtained by JSON-outputs.  

---

**Conclusion:**  
CoT yields a **minor boost** in marked-image scenarios by enforcing more structured reasoning, but this boost is inherently capped.  
Without true coordinate extraction, CoT cannot provide robust geometric accuracy beyond a few extra percentage points.  



## Baseline Pipeline: CoT-only Geometric Reasoning

This pipeline represents a **naïve baseline** where the multimodal LLM is asked to solve spatial relation questions purely through **Chain-of-Thought (CoT)** reasoning, without explicit detection or coordinate extraction.

### 1. Input
- PNG slice + free-text question  
- Example: *"Is the left kidney below the inferior vena cava?"*

### 2. Prompting with Chain-of-Thought
- The LLM is instructed to "think step by step":
  1. Identify object 1 in the image.  
  2. Identify object 2 in the image.  
  3. Compare their positions (left/right, above/below).  
  4. Output a final answer (true/false).  

Example CoT reasoning output (simplified):
- Step 1: The left kidney is usually visible on the left side of the abdomen.
- Step 2: The inferior vena cava runs along the midline.
- Step 3: The kidney appears lower than the IVC.
- Final answer: true

### 3. Direct Answer
- The model outputs a boolean prediction.

### 4. Limitations
- **No pixel-level grounding**: The LLM does not measure coordinates, it only narrates plausible relations.  
- **High variance**: Answers can change depending on prompt wording or sampling.  
- **Reliance on priors**: Often the model uses anatomical priors ("kidney is usually below liver") rather than evidence from the actual image.  
- **Accuracy**: Typically near chance level (~50–55%), showing that CoT alone is not sufficient for geometry.  

### 5. Purpose
This pipeline is **not intended as a solution**, but as a **baseline** to highlight:
- Why ungrounded CoT reasoning fails for spatial relations.  
- Why explicit coordinates (detector + geometry checker) are necessary for robust performance.

## Titles for master thesis:
- From Chain-of-Thought to Hybrid Pipelines: Robust Spatial Relation Reasoning in CT Imaging
- Parsing vs. Geometry: Hybrid Pipelines for Relation Reasoning in Medical Imaging
- Grounded visual relation verification in medical imaging via structured outputs and deterministic geometric reasoning
- 
