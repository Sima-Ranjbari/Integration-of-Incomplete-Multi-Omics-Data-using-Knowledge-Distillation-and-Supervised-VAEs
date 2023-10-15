# KD-SVAE-VCDN: Multi-Omics Integration Framework

### Overview
The rapid advancement of high-throughput biomedical technologies has led to large-scale generation of diverse omics data, including **mRNA expression**, **DNA methylation**, and **microRNA expression**. Integrating these heterogeneous datasets enables a more comprehensive understanding of the molecular basis of cancer and improves prediction of disease progression.

### Method
We propose a novel framework called **Knowledge Distillation and Supervised Variational Autoencoders with View Correlation Discovery Network (KD-SVAE-VCDN)** to integrate high-dimensional multi-omics data with limited common samples.  
The framework employs:
- **Knowledge Distillation (KD)** to transfer information between teacher and student models.  
- **Supervised Variational Autoencoders (SVAEs)** to learn latent representations of omic views.  
- **VCDN** for final multi-view fusion and classification.

**Main files:**
- `VAEs.py` – main script containing the overall KD-SVAE-VCDN architecture.  
- `KD.py` – implementation of teacher–student models for knowledge distillation.  
- `vcdn_clf.py` – classification module including the View Correlation Discovery Network (VCDN).

### Results
Experiments on breast and kidney carcinoma datasets show that **KD-SVAE-VCDN** effectively classifies patients into long- and short-term survival groups and **outperforms state-of-the-art** multi-omics integration models.

### Conclusion
This framework demonstrates the potential of multi-omics data integration for **personalized and predictive oncology**.  
By accurately forecasting disease progression at diagnosis, KD-SVAE-VCDN provides a foundation for more targeted and effective treatment strategies.
