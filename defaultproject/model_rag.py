
from utils.etc import hit2docdict
import torch
# Modify
class ModelRAG():
    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    def set_retriever(self, retriever):
        self.retriever = retriever

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def search(self, queries, qids, k=5):
        # Use the retriever to get relevant documents
        list_passages = []
        list_scores = []

        # fill here
        ######
        
        #### YOUR CODES; TODO 
        qids = [str(qid) for qid in qids]
        # BM25 top-k for every query in the batch: {qid: [hit, ...]}
        batch_hits = self.retriever.batch_search(queries=list(queries), qids=qids, k=k, threads=8)
        for qid in qids:
            passages = []
            scores = []
            for hit in batch_hits[qid]:
                doc = hit2docdict(hit)  # {"id": ..., "contents": '"<title>"\n<passage text>'}
                title, sep, text = doc["contents"].partition("\n")
                if not sep:
                    title, text = "", title
                passages.append({"id": doc.get("id", hit.docid), "title": title.strip().strip('"'), "text": text.strip()})
                scores.append(hit.score)
            list_passages.append(passages)
            list_scores.append(scores)
        ######

        return list_passages, list_scores

    # Modify
    def make_augmented_inputs_for_generate(self, queries, qids, k=5):
        # Get the relevant documents for each query
        list_passages, list_scores = self.search(queries, qids, k=k)
        
        list_input_text_without_answer = []
        # fill here
        ######
        
        #### YOUR CODES; TODO 
        # same format as RAGDataset (training): "Title: ..\nPassage: ..\n...\nQuestion: ..\nAnswer:"
        for query, passages in zip(queries, list_passages):
            input_text_ctx = "\n".join(f"Title: {p['title']}\nPassage: {p['text']}" for p in passages)
            ctx_ids = self.tokenizer(input_text_ctx, add_special_tokens=False)["input_ids"]
            if len(ctx_ids) > 768:  # same context budget as RAGDataset, keeps GPT-small inside its 1024 window
                input_text_ctx = self.tokenizer.decode(ctx_ids[:768], clean_up_tokenization_spaces=False)
            list_input_text_without_answer.append(f"{input_text_ctx}\nQuestion: {query}\nAnswer:")
        ######
        
        return list_input_text_without_answer

    @torch.no_grad()
    def retrieval_augmented_generate(self, queries, qids,k=5, **kwargs):
        # fill here:
        ######
        
        #### YOUR CODES; TODO 
        input_texts = self.make_augmented_inputs_for_generate(queries, qids, k=k)
        self.tokenizer.padding_side = "left"  # decoder-only generation continues from the right end
        inputs = self.tokenizer(input_texts, padding="longest", return_tensors="pt", return_token_type_ids=False)
        ######

        # # Move batch to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            **kwargs
        )
        
        outputs = outputs[:, inputs['input_ids'].size(1):]

        return outputs
