"""The `dig_order` pipeline with the source-citing output contract.

Retrieval is left bit-identical to `dig_order` -- same BM25 pool, same RRF of UPR
and the answer-likelihood proxy, same slot order -- so that HasAnswer cannot move
and any change in accuracy is attributable to the reader. Only the prompt the
reader sees and the way its output is parsed differ.

`adapter_path=None` is the G1 control: the untrained model under the new
contract. It is the comparison the training result has to beat, because the
contract change alone also moves the number.
"""
from improved_rag import ImprovedRAG
from reader_rft import (SLOTS, SYS_SOURCE, build_messages,
                        format_numbered_passages, parse_output)


class SourceReaderRAG(ImprovedRAG):
    """ImprovedRAG whose reader names the passage it used."""

    def __init__(self, adapter_path=None, max_new_tokens=40, num_shots=2, **kwargs):
        kwargs.setdefault("variant", "v3")
        super().__init__(**kwargs)
        self.system_prompt = SYS_SOURCE
        self.num_shots = num_shots
        self.max_new_tokens = max_new_tokens
        self.adapter_path = adapter_path
        self._adapter_applied = False
        # per-question records, consumed by the analysis scripts
        self.source_records = []

    # -- prompt --------------------------------------------------------------

    def format_passages(self, passages):
        return format_numbered_passages(passages)

    def build_messages(self, question, passages):
        # delegated so the rollout sampler and the reader cannot diverge
        return build_messages(question, passages, self.num_shots)

    # -- model ---------------------------------------------------------------

    def set_model(self, model):
        """Attach the LoRA adapter once, to the reader itself.

        Unlike the selector experiment there is no second model here: the thing
        being trained *is* the reader, so the adapter belongs on `self.model`.
        """
        super().set_model(model)
        if self.adapter_path and not self._adapter_applied:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path).eval()
            self._adapter_applied = True

    # -- parsing -------------------------------------------------------------

    def postprocess(self, predictions):
        """Record source and validity per question, return answers for scoring.

        Overrides the PARSERS dispatch: this contract has a second field, and the
        existing json parser would discard it.
        """
        answers = []
        valid_n = 0
        for text in predictions:
            answer, source, valid = parse_output(text, SLOTS)
            self.source_records.append({"raw": text, "answer": answer,
                                        "source": source, "valid": valid})
            valid_n += int(valid)
            answers.append(answer)
        self.parse_stats = {
            "n": len(predictions),
            "valid_format_rate": valid_n / len(predictions) if predictions else 0.0,
            "mean_pred_words": (sum(len(a.split()) for a in answers) / len(answers)
                                if answers else 0.0),
            "pct_empty": (sum(1 for a in answers if not a.strip()) / len(answers)
                          if answers else 0.0),
        }
        return answers

    @property
    def assembly_stats(self):
        total = len(self.source_records) or 1
        valid = sum(1 for r in self.source_records if r["valid"])
        base = {"adapter": self.adapter_path or "none",
                "parsed": len(self.source_records),
                "valid_format_rate": valid / total}
        parent = super().assembly_stats
        if isinstance(parent, dict):
            base.update({k: v for k, v in parent.items() if k not in base})
        return base
