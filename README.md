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
Problem: Detector needs to be trained again on ct scans. For a final, research-grade workflow → you need a full 3D detector/segmenter on DICOM/NIfTI.  
Training a PNG model is only worth it for short-term prototypes, not for serious research.  



