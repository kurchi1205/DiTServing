class SDTokenizer:
    def __init__(
        self,
        max_length=77,
        pad_with_end=True,
        tokenizer=None,
        has_start_token=True,
        pad_to_max_length=True,
        min_length=None,
        extra_padding_token=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length

        empty = self.tokenizer("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        self.extra_padding_token = extra_padding_token

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8

    def tokenize_with_weights(self, text: str, return_word_ids=False):
        """
        Tokenize the text, with weight values - presume 1.0 for all and ignore other features here.
        The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3.
        """
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        # tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = (
                unescape_important(weighted_segment).replace("\n", " ").split(" ")
            )
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # parse word
                tokens.append(
                    [
                        (t, weight)
                        for t in self.tokenizer(word)["input_ids"][
                            self.tokens_start : -1
                        ]
                    ]
                )

        # reshape token array to CLIP input size
        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            # determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    # break word in two and add end token
                    if is_large:
                        batch.extend(
                            [(t, w, i + 1) for t, w in t_group[:remaining_length]]
                        )
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    # add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    # start new batch
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t, w, i + 1) for t, w in t_group])
                    t_group = []

        # pad extra padding token first befor getting to the end token
        if self.extra_padding_token is not None:
            batch.extend(
                [(self.extra_padding_token, 1.0, 0)]
                * (self.min_length - len(batch) - 1)
            )
        # fill last batch
        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.min_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]

        return batched_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)

class T5XXLTokenizer(SDTokenizer):
    """Wraps the T5 Tokenizer from HF into the SDTokenizer interface"""

    def __init__(self):
        super().__init__(
            pad_with_end=False,
            tokenizer=T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl"),
            has_start_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=77,
        )