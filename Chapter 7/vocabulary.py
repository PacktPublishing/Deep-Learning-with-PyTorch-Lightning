class Vocabulary(object):
    def __init__(self):
        self.token_to_int = {}
        self.int_to_token = {}
        self.current_index = 0

    def __call__(self, token):
        if not token in self.token_to_int:
            return self.token_to_int['<unk>']
        return self.token_to_int[token]

    def __len__(self):
        return len(self.token_to_int)

    def add_token(self, token):
        if not token in self.token_to_int:
            self.token_to_int[token] = self.current_index
            self.int_to_token[self.current_index] = token
            self.current_index += 1
