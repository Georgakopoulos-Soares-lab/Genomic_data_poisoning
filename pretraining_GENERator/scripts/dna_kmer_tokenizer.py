import itertools
import os
import json
import re
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer

class DNAKmerTokenizer(PreTrainedTokenizer):
    def __init__(self, k, **kwargs):
        self.k = k
        self.special_tokens = [
            "<oov>",
            "<s>",
            "</s>",
            "<pad>",
            "<mask>",
            "<bog>",
            "<eog>",
            "<bok>",
            "<eok>",
            "<+>",
            "<->",
            "<cds>",
            "<pseudo>",
            "<tRNA>",
            "<rRNA>",
            "<ncRNA>",
            "<miscRNA>",
            "<mam>",
            "<vrt>",
            "<inv>",
            "<pln>",
            "<fng>",
            "<prt>",
            "<arc>",
            "<bct>",
            "<mit>",
            "<plt>",
            "<plm>",
            "<vir>",
            "<sp0>",
            "<sp1>",
            "<sp2>",
        ]
        self.kmers = [
            "".join(kmer) for kmer in itertools.product("ATCG", repeat=self.k)
        ]
        self.vocab = {
            token: i for i, token in enumerate(self.special_tokens + self.kmers)
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.special_token_pattern = re.compile(
            "|".join(re.escape(token) for token in self.special_tokens)
        )
        self.dna_pattern = re.compile(f"[A-Z]{{{self.k}}}|[A-Z]+")
        # Must call super().__init__() before setting special token attributes
        # to ensure _special_tokens_map exists (required by transformers >= 4.46)
        kwargs["bos_token"] = "<s>"
        kwargs["eos_token"] = "</s>"
        super().__init__(**kwargs)
        self._bos_token_id = self._convert_token_to_id("<s>")
        self._eos_token_id = self._convert_token_to_id("</s>")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text, **kwargs) -> List[str]:
        tokens = []
        pos = 0
        while pos < len(text):
            special_match = self.special_token_pattern.match(text, pos)
            if special_match:
                tokens.append(special_match.group())
                pos = special_match.end()
            else:
                dna_match = self.dna_pattern.match(text, pos)
                if dna_match:
                    dna_seq = dna_match.group()
                    tokens.append(dna_seq)
                    pos = dna_match.end()
                else:
                    tokens.append(text[pos])
                    pos += 1
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["<oov>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, "<oov>")

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
            self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def prepare_for_model(self, *args, **kwargs):
        encoding = super().prepare_for_model(*args, **kwargs)
        if "token_type_ids" in encoding:
            del encoding["token_type_ids"]
        return encoding

    def save_vocabulary(
            self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        import os

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
        )
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                writer.write(token + "\n")
        return (vocab_file,)
    
    def save_pretrained(self, save_directory: str, **kwargs):
        vocab_files = super().save_pretrained(save_directory, **kwargs)
        tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")
        
        # 读取现有的配置或创建新的
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}
        
        # 添加auto_map配置
        config.update({
            "auto_map": {
                "AutoTokenizer": [
                    "tokenizer.DNAKmerTokenizer",
                    None
                ]
            },
        })
        
        # 添加kmer配置
        config.update({
            "k": self.k
        })
        
        # 保存配置
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return vocab_files
