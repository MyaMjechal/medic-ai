# Medic AI: AI for Your Medical Needs

## Class Project for AT82.05 – Artificial Intelligence: Natural Language Understanding  
**Supervised by:** Asst. Prof. Chaklam Silpasuwanchai  
**Team Name:** Semantic Bard  

## Table of Contents

- [Team Members](#team-members)
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Motivation](#motivation)
- [Use Cases & Expected Results](#use-cases--expected-results)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Selection](#model-selection)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Progress](#progress)
- [License](#license)
- [References](#references)

## Team Members

- **Htet Wai Yan Htut** – Department of Data Sciences and Artificial Intelligence, AIT ([st125326@ait.ac.th](mailto:st125326@ait.ac.th))  
- **Zwe Htet** – Department of Data Sciences and Artificial Intelligence, AIT ([st125338@ait.ac.th](mailto:st125338@ait.ac.th))  
- **Mya Mjechal** – Department of Data Sciences and Artificial Intelligence, AIT ([st125469@ait.ac.th](mailto:st125469@ait.ac.th))  

**Medical Advisor:** Dr. Khin Lay Phyu (M.B.,B.S)

## Introduction

**Medic AI** is a multimodal AI-powered mobile application designed to enhance healthcare accessibility, medication safety, and emergency response. By leveraging Natural Language Processing (NLP), Computer Vision (CV), and multimodal AI, Medic AI provides three key functionalities:

1. **Medicine Scanning** – Extracts and summarizes medication details from packaging images.
2. **Specialist Recommendation** – Classifies symptoms to recommend appropriate medical specialists.
3. **Emergency Assistance** – Generates actionable summaries from emergency scene images and GPS data.

The project benchmarks Medic AI’s performance against medical databases, Optical Character Recognition (OCR)/NLP techniques, and expert-generated emergency descriptions, ensuring clinical validation through our **medical advisor**.

## Problem Statement

### Key Challenges in Healthcare:
1. **Limited Access to Medication Information:** Patients often struggle to interpret complex medication labels or lack immediate access to doctors for clarifications, increasing risks of incorrect usage and preventable harm.
2. **Inefficient Specialist Referrals:** Individuals relying on hospital call centers or general practitioners for specialist recommendations frequently face mismatches, delaying diagnosis and treatment for time-sensitive conditions.
3. **Incomplete Emergency Context:** Traditional emergency calls depend heavily on verbal descriptions, which may lack critical visual or contextual details (e.g., accident scenes, victim conditions), slowing first responders’ ability to prioritize and act.

## Motivation

Medic AI addresses the mentioned problem statements above by integrating multimodal AI (NLP and computer vision) into a single, user-centric platform. 
- **Medication Insights:** Extract and summarize medication details directly from packaging images, eliminating dependence on costly consultations (Use Case 1). 
- **Specialist Recommendation:** Classify symptoms via NLP to recommend the most relevant specialist, reducing referral delays (Use Case 2). 
- **Medical Emergency:** Generate actionable summaries of emergency scenes using image analysis and GPS data, bridging the gap between bystanders and first responders (Use Case 3). 

Our team consists of:
- **3 Master’s Students:** Expertise in NLP, computer vision, and multimodal AI systems. 
- **Medical Advisor:** A practicing physician ensuring clinical validity and ethical alignment.

Our team wants to give quicker and reliable access to modern healthcare to everyone. By merging AI with real-world medical insights, Medic AI aims to make quality and modern healthcare accessible to everyone with an internet and empower normal people to make informed, timely medical decisions. 

## Use Cases & Expected Results

### **Use Case 1: Medication Insights**
Patients often struggle with understanding complex medication labels, leading to incorrect usage. Medic AI provides a solution by allowing users to take a picture of the medicine packaging to extract and summarize critical details.

**Workflow:**
1. **Image Capture:** User uploads a medicine packaging image.
2. **OCR Processing:** Extracts text from the label.
3. **NLP Cleanup:** Removes irrelevant text (e.g., logos, marketing).
4. **Entity Recognition:** Identifies drug name, dosage, and warnings.
5. **Summary Generation:** Outputs a concise medication overview.

**Expected Outcome:**  
Users receive clear, AI-generated medication summaries, reducing the risk of misuse.

### **Use Case 2: Specialist Recommendation**
Finding the right medical specialist can be challenging, leading to delayed diagnoses. Medic AI helps by analyzing user-reported symptoms and recommending the most relevant specialist.

**Workflow:**
1. **Symptom Input:** Users describe symptoms and select the affected body part.
2. **NLP Processing:** Classifies symptoms using intent recognition.
3. **Specialist Mapping:** Matches symptoms to the appropriate specialist.
4. **Recommendation Output:** Provides specialist suggestion with justification.

**Expected Outcome:**  
Users receive an AI-driven specialist recommendation, improving healthcare access and accuracy.

### **Use Case 3: Medical Emergency**
Emergency responders often rely on incomplete verbal descriptions. Medic AI enhances emergency response by analyzing uploaded images and generating actionable summaries.

**Workflow:**
1. **Emergency Image Upload:** Users capture an accident/emergency scene.
2. **Multimodal AI Analysis:** Generates a textual description of the scene.
3. **Scene Understanding:** Classifies severity using AI models.
4. **Emergency Dispatch:** Sends AI summary, image, and location to emergency services.

**Expected Outcome:**  
First responders gain critical visual and contextual information, improving response time and prioritization.

## Related Work
### **A. Medicine Scan and Information Extraction**  

#### **ViLMedic: A Framework for Research at the Intersection of Vision and Language in Medical AI**  
- **Authors:** Jean-benoit Delbrouck, Saab, K., Varma, M., Sabri Eyuboglu, Chambon, P., Dunnmon, J., Zambrano, J., Chaudhari, A., & Langlotz, C.  
- **Year:** 2022
- **Summary:** ViLMedic presents a **vision-language framework** for medical AI, demonstrating **OCR-based text extraction from medical packaging** and **structured summarization using NLP models**. The study validates the effectiveness of **multimodal learning** in **medical document processing**, aligning with Medic AI’s **medicine scanning module**.  

#### **LlamaCare: An Instruction Fine-Tuned Large Language Model for Clinical NLP**  
- **Authors:** Li, R., Wang, X., & Yu, H.   
- **Year:** 2024
- **Summary:** LlamaCare fine-tunes **large clinical NLP models** using instruction-based learning, significantly improving their ability to **extract structured medical information**. The study highlights how **domain-specific instructions enhance clinical text coherence and accuracy**, making it relevant for **Medic AI’s medication summarization feature**.  

### **B. Specialist Recommendation Based on Symptoms**  

#### **Medical Knowledge-Enhanced Prompt Learning for Diagnosis Classification from Clinical Text**  
- **Authors:** Lu, Y., Zhao, X., & Wang, J.   
- **Year:** 2023
- **Summary:** This study introduces a **medical knowledge-enhanced prompting framework (MedKPL)** for **symptom classification and diagnostic reasoning**. By leveraging domain-specific knowledge, the model significantly improves **accuracy in symptom-to-specialist mapping**, aligning with Medic AI’s **specialist recommendation module**.  

### **C. Visual Emergency Reporting**  

#### **ERVQA: A Dataset to Benchmark the Readiness of Large Vision Language Models in Hospital Environments**  
- **Authors:** Ray, S., Gupta, K., Kundu, S., Dr. Payal Arvind Kasat, Somak Aditya, & Goyal, P.   
- **Year:** 2024
- **Summary:** ERVQA introduces a **benchmark dataset for evaluating vision-language models in medical emergency scenarios**. The dataset enables **AI-driven emergency scene understanding**, validating Medic AI’s **image captioning and accident severity classification model**.  

#### **Benchmarking Large Language Models on Communicative Medical Coaching: A Dataset and a Novel System**  
- **Authors:** Huang, H., Wang, S., Liu, H., Wang, H., & Wang, Y.   
- **Year:** 2024
- **Summary:** This paper evaluates **large language models in real-time medical interactions**, highlighting their potential in **AI-assisted emergency response**. The study underscores how **GPT-based multimodal AI models** can generate **accurate emergency scene descriptions**, contributing to **Medic AI’s emergency assistance module**.  

#### **MUMOSA: Interactive Dashboard for Multi-Modal Situation Awareness**  
- **Authors:** Lukin, S. M., Bowser, S., Suchocki, R., Summers-Stay, D., Ferraro, F., Matuszek, C., & Voss, C.
- **Year:** 2024  
- **Summary:** MUMOSA explores **multi-modal situation awareness**, showcasing the ability of **AI to generate real-time, context-aware emergency scene summaries**. This aligns with **Medic AI’s emergency dispatch feature**, where **AI-generated summaries assist first responders**.  

## Methodology
![medic_ai_proposal drawio (1)](https://github.com/user-attachments/assets/a4c068ae-b5ca-426c-bdb5-19c3d0f402b6)

### **Dataset**  
To ensure comprehensive training and evaluation, Medic AI utilizes publicly available medical repositories and proprietary datasets. These datasets facilitate robust AI model development by covering real-world healthcare scenarios.  

- **Medication Scanning Dataset:** A collection of high-resolution images of medicine packaging with corresponding textual information detailing drug usage, dosage, and side effects. This dataset is critical for training the **OCR model**. Data acquisition is currently in progress, with an expected response time of 5 to 10 days.  
- **Emergency Assistance Dataset:** A dataset consisting of images depicting medical emergencies (e.g., injuries, accidents), each paired with expert-generated textual descriptions. The dataset is under request and is expected to be available alongside the medication dataset.  
- **Specialist Recommendation Dataset:** The **MIMIC-IV dataset**, a widely used repository of **clinical notes and patient records**, is utilized to extract **symptom descriptions** and corresponding **specialist recommendations**. Data acquisition is currently under their review and waiting for approval.

These datasets enable **Medic AI** to **train on real-world medical scenarios**, enhancing its ability to deliver **accurate and contextually relevant predictions**.  

### **Data Preprocessing**  
Preprocessing ensures the input data is optimized for **AI model performance**, improving accuracy while reducing noise and inconsistencies.  

- **Medication Scanning:**  
  - Image preprocessing involves **resizing, normalization, and binarization** to improve OCR text readability.  
  - Extracted text undergoes **cleaning, OCR error correction, and structuring** to retain **medically relevant details**.  

- **Specialist Recommendation:**  
  - Text preprocessing includes **tokenization, stop-word removal, and lemmatization** to standardize medical terms.  
  - **Named Entity Recognition (NER)** is employed to extract **key symptoms** and classify them under **appropriate medical categories**.  

- **Emergency Assistance:**  
  - Image preprocessing includes **resizing and normalization** for consistent input.  
  - Text descriptions are **tokenized and structured** for multimodal model training.  

These steps **improve the quality of input data**, ensuring **high accuracy in model predictions**.

### **Model Selection**  
Medic AI employs **state-of-the-art AI models**, each optimized for its respective task.  

- **Medication Scanning:**  
  - Uses a deep learning-based **OCR model** such as **Google Vision** and **Tesseract** to extract text.  
  - A **knowledge-based query system** retrieves structured drug information from **DrugBank**.  
  - A transformer-based **NLP model (LSTM, BERT)** refines and summarizes extracted details.  

- **Specialist Recommendation:**  
  - A **fine-tuned BioGPT model** developed by Microsoft, trained on **MIMIC-IV** for **symptom classification**.  
  - The model maps symptom descriptions to **specialties**, ensuring high-precision **specialist recommendations**.  

- **Emergency Assistance:**  
  - **GPT-4 Vision** is used for **image feature extraction**, combined with a **CNN-based YOLOv8 model** for **real-time object detection**.  
  - A **multimodal AI approach** similar to **medical image captioning models** is employed to generate meaningful **emergency scene descriptions**.  

Each model undergoes **rigorous training and validation** to maintain **robust performance**.  

### **Training Strategy**  
Medic AI’s AI models are trained using **large-scale datasets** and optimized through **iterative learning processes**.  

- **Medication Scanning:**  
  - OCR models are trained on **medicine packaging images** with **ground truth annotations**.  
  - The **knowledge base system** is preloaded with **structured data from DrugBank**, eliminating additional training needs.  

- **Specialist Recommendation:**  
  - The **classification model** is fine-tuned on the **MIMIC-IV dataset** using **cross-entropy loss** and optimized with the **Adam optimizer**.  
  - **Cross-validation techniques** ensure **generalization to unseen cases**.  

- **Emergency Assistance:**  
  - The **multimodal model** is trained on **emergency images** with **expert-generated textual descriptions**.  
  - Loss functions such as **cross-entropy** optimize performance for **image captioning tasks**.  
  - Regular **validation steps** ensure **real-world applicability**.  

### **Evaluation Metrics**  
Medic AI's models are evaluated using **industry-standard metrics**, ensuring **objective performance assessment and continuous refinement**.  

- **Medication Scanning:**  
  - **OCR accuracy** is measured using **character and word-level recognition metrics**.  
  - **BLEU scores** and **expert validation** assess the quality of generated summaries.  

- **Specialist Recommendation:**  
  - Evaluated using **accuracy, precision, recall, and F1-score**.  
  - Benchmark comparisons against **Med-PaLM** and **BioGPT** validate model performance.  

- **Emergency Assistance:**  
  - Evaluated using **CIDEr, BLEU, and ROUGE** to measure **text quality** in emergency descriptions.  
  - The **generated text** is compared against **expert-labeled emergency scene descriptions** to ensure clinical relevance.  

These metrics provide a **comprehensive understanding** of **Medic AI’s effectiveness**, guiding **further improvements and refinements**.  

## Progress  

### **Current Achievements**  
- [x] **Team Formation:** Three AI experts and one medical expert assembled.  
- [x] **Project Scope Defined:** Three key use cases established.  
- [x] **Datasets & Models Identified:**  
  - **Medical Databases:** DrugBank, MIMIC-IV  
  - **OCR Tools:** Google Vision, Tesseract  
  - **NLP Models:** LSTM, BERT  
  - **Vision-Language Models:** YOLOv8, GPT-4 Vision  
  - **Medical Reasoning Model:** BioGPT (Microsoft)  
- [x] **Research Questions Identified:**  
  - How effectively can **Medic AI** summarize and extract key medication information compared to standard medical databases?  
  - What **NLP and image recognition techniques** provide the highest accuracy and speed for structured medical detail extraction?  
  - How accurately can **Medic AI classify and recommend medical specialists** using expert-validated LLMs (e.g., Med-PaLM, BioGPT)?  
  - How well can **Medic AI’s multimodal AI model** describe emergency situations and assess victim conditions?  

### **Next Steps**  
- [ ] **Dataset & Model Exploration:** Further investigation into identified datasets and model selection.  
- [ ] **Proposal Revision:** Incorporate feedback from the professor/TA.  
- [ ] **System Development:** Begin implementation based on the revised proposal and project scope.  

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

## References

1. Delbrouck, J.-B., Saab, K., Varma, M., Eyuboglu, S., Chambon, P., Dunnmon, J., Zambrano, J., Chaudhari, A., & Langlotz, C. (2022). *ViLMedic: A framework for research at the intersection of vision and language in medical AI*. [ACL Demo](https://doi.org/10.18653/v1/2022.acl-demo.3).
2. Li, R., Wang, X., & Yu, H. (2024). *LlamaCare: An Instruction Fine-Tuned Large Language Model for Clinical NLP*. [ACL Anthology](https://aclanthology.org/2024.lrec-main.930/).
3. Lu, Y., Zhao, X., & Wang, J. (2023). *Medical knowledge-enhanced prompt learning for diagnosis classification from clinical text*. [Clinical NLP](https://doi.org/10.18653/v1/2023.clinicalnlp-1.33).
4. Ray, S., Gupta, K., Kundu, S., Kasat, P. A., Aditya, S., & Goyal, P. (2024). *ERVQA: A Dataset to Benchmark the Readiness of Large Vision Language Models in Hospital Environments*. [EMNLP](https://doi.org/10.18653/v1/2024.emnlp-main.873).
5. Huang, H., Wang, S., Liu, H., Wang, H., & Wang, Y. (2024). *Benchmarking Large Language Models on Communicative Medical Coaching: A Dataset and a Novel System*. [ACL Findings](https://doi.org/10.18653/v1/2024.findings-acl.94).
6. Lukin, S. M., Bowser, S., Suchocki, R., Summers-Stay, D., Ferraro, F., Matuszek, C., & Voss, C. (2024). *MUMOSA, Interactive Dashboard for Multi-Modal Situation Awareness*. [Future Directions in AI](https://doi.org/10.18653/v1/2024.futured-1.4).
