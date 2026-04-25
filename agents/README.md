<div align="center">
 
![logo](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/assets/logo_rounded.png)

<h1 align="center"><strong>🤖 Agent Details :<h6 align="center">All implemented agents have been detailed below</h6></strong></h1>

</div>

---
 
## 📚 Table of Contents
- [Computer Vision Agents (heyi-Trans-master Integration)](#cv-agents)
- [Human-in-the-loop Validation Agent](#human-in-the-loop)
- [Research-papers-and-documents-used-for-RAG-Citations](#citations)

---

## 📌 Computer Vision Agents — `heyi-Trans-master` Integration <a name="cv-agents"></a>

Both brain-imaging agents (`BRAIN_TUMOR_AGENT`, `BRAIN_STROKE_AGENT`) are now wired up to the
in-repo general-purpose Vision Transformer framework [`heyi-Trans-master`](../heyi-Trans-master/README.md).
The three-stage reserved interface (`segment_image` → `mark_lesion` → `diagnose`) is fully implemented;
`IMPLEMENTED = True` on both agent classes.

### Module layout

```
agents/image_analysis_agent/
├── heyi_adapter.py                        ← Bridge to heyi-Trans-master (ViT encoder + binary seg decoder)
├── image_classifier.py                    ← GLM-4V based router (is this a brain MRI / CT?)
├── brain_tumor_agent/
│   ├── brain_tumor_inference.py           ← BrainTumorAgent (IMPLEMENTED = True)
│   └── models/                            ← Drop `brain_tumor_segmentation.pth` here
└── brain_stroke_agent/
    ├── brain_stroke_inference.py          ← BrainStrokeAgent (IMPLEMENTED = True)
    └── models/                            ← Drop `brain_stroke_segmentation.pth` here
```

### Per-stage behaviour

| Stage | Implementation |
|-------|----------------|
| `segment_image` | Loads image → resize 224×224 → `ViTEncoder` (from `heyi-Trans-master`) → `patch_features (B,196,768)` → lightweight decoder (`LayerNorm + Linear + GELU + Linear`) → bilinear upsample to original resolution → binary mask `[H,W]` |
| `mark_lesion` | Semi-transparent color overlay on the lesion region + yellow contour outline; saved to `uploads/brain_{tumor,stroke}_output/*.png` (already static-mounted by `app.py`) |
| `diagnose` | Uses mask statistics (region count, area ratio, centroid location, bounding box) to produce a structured Chinese diagnosis text. Adds an explicit ⚠️ *demo-mode warning* if no fine-tuned weights are loaded, so the output is never mistaken for clinical advice. |

### Weights

- File names are **fixed** by `config.py → MedicalCVConfig`:
  - `agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth`
  - `agents/image_analysis_agent/brain_stroke_agent/models/brain_stroke_segmentation.pth`
- No weights = **Demo mode** (ImageNet-pretrained ViT features + untuned head). End-to-end pipeline runs, but segmentation is noisy.
- With compatible weights = **Production mode**. Adapter supports 4 state-dict formats automatically; see the top-level README for details.

### Data flow (end to end)

```
User uploads image
  → POST /upload  (app.py)
  → process_query (agent_decision.py)
    → ImageClassifier (GLM-4V determines if brain MRI / CT)
    → BRAIN_TUMOR_AGENT / BRAIN_STROKE_AGENT node
      → config.image_analyzer.detect_brain_tumor(image_path)
        → BrainTumorAgent.predict()
          → segment_image()  → HeyiVisionAdapter.infer_mask()
          → mark_lesion()    → HeyiVisionAdapter.overlay()  → uploads/brain_tumor_output/brain_tumor_plot.png
          → diagnose()       → Chinese diagnosis text with optional demo-mode warning
    → Frontend shows { response: text, result_image: url } (rendered side-by-side with the original image)
    → Human-in-the-Loop validation gate (see below)
```

---

## 📌 Human-in-the-loop validation of Medical Computer Vision Diagnosis Agents' Outputs <a name="human-in-the-loop"></a>

In `agent_decision.py`:

1. Interrupt the workflow when human validation is needed
2. Store the interrupted state in memory
3. Add endpoints to expose pending validations and submit validation decisions
4. Resume the workflow after the human has provided feedback

On frontend:

1. Check if a response needs validation (needs_validation flag)
2. If so, show a validation interface to the human reviewer
3. Send the validation decision back through the /validate endpoint
4. Continue the conversation

Implemented a complete human-in-the-loop validation system using LangGraph's NodeInterrupt functionality, integrated with the backend and frontend.

---

> [!NOTE]
> More details about other agents to be added.

---

## 📌 Research Papers and Documents Used for RAG (Citations) <a name="citations"></a>

1. Saeedi, S., Rezayi, S., Keshavarz, H. et al. MRI-based brain tumor detection using convolutional deep learning methods and chosen machine learning techniques. BMC Med Inform Decis Mak 23, 16 (2023). [https://doi.org/10.1186/s12911-023-02114-6](https://doi.org/10.1186/s12911-023-02114-6)

2. Babu Vimala, B., Srinivasan, S., Mathivanan, S.K. et al. Detection and classification of brain tumor using hybrid deep learning models. Sci Rep 13, 23029 (2023). [https://doi.org/10.1038/s41598-023-50505-6](https://doi.org/10.1038/s41598-023-50505-6)

3. Khaliki, M.Z., Başarslan, M.S. Brain tumor detection from images and comparison with transfer learning methods and 3-layer CNN. Sci Rep 14, 2664 (2024). [https://doi.org/10.1038/s41598-024-52823-9](https://doi.org/10.1038/s41598-024-52823-9)

4. Brain Tumors: an Introduction basic level, Mayfield Clinic, UCNI

---

