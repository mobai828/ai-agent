<div align="center">
 
![logo](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/assets/logo_rounded.png)

<h1 align="center"><strong>🤖 Agent Details :<h6 align="center">All implemented agents have been detailed below</h6></strong></h1>

</div>

---
 
## 📚 Table of Contents
- [Human-in-the-loop Validation Agent](#human-in-the-loop)
- [Research-papers-and-documents-used-for-RAG-Citations](#citations)

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

