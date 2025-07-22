# Hybrid-Legal-Search-Engine
**Combining BM25 with Domain-Adapted LegalBERT**

The system employs BM25 for the initial retrieval of documents, followed by semantic re-ranking using a
lightweight, domain-adapted version of BERT (LEGALBERTSMALL). By integrating these two
approaches, our design aims to address the challenges of legal text retrieval, offering improved
relevance and efficiency for complex legal queries.


## Why a Hybrid Legal Search Engine?

Legal documents contain:  
- Formal, domain-specific vocabulary  
- Dense references (e.g., case laws, statutes)  

**BM25** is:  
- Fast and interpretable  
- Lacks semantic understanding  

**LEGALBERTSMALL** is:  
- Domain-aware and context-sensitive  
- Computationally intensive on large corpora  

**Problem:** Neither approach alone suffices for real-world legal research.  

**Our Solution:** Combine BM25’s speed with LEGALBERT’s semantic depth.  

---

## System Architecture

![image](https://github.com/user-attachments/assets/09cc5eb5-6bb6-48e6-9328-4f5c1faf5645)

1. **Query Handling**  
   User submits a legal query via the UI (FastAPI backend)  

2. **Initial Retrieval**  
   Query is sent to Elasticsearch  
   BM25 retrieves top *K* candidate documents (e.g., Top 100)  

3. **Semantic Re-Ranking**  
   LEGALBERTSMALL scores and re-ranks the candidates  
   Outputs top *N* most relevant documents (e.g., Top 10–20)  

4. **Response**  
   Re-ranked results are returned to the user interface for exploration

![image](https://github.com/user-attachments/assets/5d03ebcd-efca-46f3-9f3a-485715df3116)


---

## Implementation Stack

![image](https://github.com/user-attachments/assets/f9737119-c40b-401c-acba-51c12614a7ce)


- FastAPI (Backend & API)
- Elasticsearch (BM25-based retrieval)
- LEGALBERTSMALL (Semantic re-ranking)
- CASE_HOLD dataset (Evaluation and testing)

---

## Example Query Output

**Query Submitted:**  
> “data protection”

**System Response:**  
*Results are from the CASE_HOLD dataset and reflect LEGALBERT re-ranking. All results shown are re-ranked — no raw BM25 output is exposed.*

![image](https://github.com/user-attachments/assets/74cfb4a3-eefd-4e4b-a31e-8f7c4788e8b4)


---

## Qualitative Evaluation & Observations

- Current implementation outputs **only final ranked results**  
- No ground-truth labels or automatic relevance metrics used (yet)  

---

## Key Strengths & Future Plans

- **Shallow keyword search:** LEGALBERTSMALL for semantic re-ranking  
- **Complex legal vocabulary:** Domain-adapted transformer embeddings  
- **Speed vs accuracy trade-off:** BM25 for speed + LEGALBERT for precision  
- **Usability:** Simple interface with fast, interpretable results  

### Future Plans:
- Introduce measurable evaluations  
- Add explainability features  
- Expand to new legal domains and languages  

---
