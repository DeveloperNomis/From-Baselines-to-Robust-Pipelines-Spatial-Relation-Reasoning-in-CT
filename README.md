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



