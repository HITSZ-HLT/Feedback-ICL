from transformers import AutoModelForCausalLM
import torch

class LLMgenerate():
    def __init__(self, cfg, tokenizer, all_labels=None):
        self.cfg = cfg
        precison = self._init_precision(cfg.load_bit)
        if cfg.device == 'cuda':
            # multiple GPUs
            print("Using Multiple GPUs for infering")
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, device_map="auto", **precison)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, device_map=cfg.device, **precison)
        self.tokenizer = tokenizer
        if all_labels is not None:
            self.all_labels = all_labels
            self.verbalizer = self.get_verbalizer(all_labels)

    def _init_precision(self, load_bit):
        if load_bit == "fp16":
            precision = {"torch_dtype": torch.float16}
        elif load_bit =="bf16":
            precision = {"torch_dtype": torch.bfloat16}
        elif load_bit == "fp32":
            precision = {"torch_dtype": torch.float32}
        return precision

    def get_verbalizer(self, labels):
        v = []
        for label in labels:
            token_ids = self.tokenizer.tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(token_ids) > 1:
                print(f"Truncation token ids: {label} -- {list(token_ids)}")
            v.append(token_ids[0])
        return v
            
    
    def generate_decode(self, **kwargs):
        generated_ids = self.model.generate(**kwargs, pad_token_id=self.tokenizer.tokenizer.eos_token_id)
        generated_ids = generated_ids[:, kwargs['input_ids'].shape[1]:]
        output = self.tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output
    
    def generate_cls(self, **kwargs):
        generated_ids = self.model.generate(**kwargs, pad_token_id=self.tokenizer.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True)
        scores = generated_ids.scores[0].cpu()
        # find corresponding token id in verbalizer
        scores = scores[:, self.verbalizer]
        assert scores.shape[1] == len(self.all_labels)
        # random if scores == -inf; or use 'others' label
        for i in range(scores.shape[0]):
            if scores[i].max() == scores[i].min():
                if "others" in self.all_labels:
                    scores[i][self.all_labels.index("others")] = 1.0
                else:
                    scores[i] = torch.rand_like(scores[i])
        scores = scores.argmax(-1)
        # get labels
        output = [self.all_labels[x] for x in list(scores)]
        return output
        