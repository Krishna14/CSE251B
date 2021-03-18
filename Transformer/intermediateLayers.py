class Embedder(nn.Module):
    """ Used to store the embedding """
    def __init__(self, vocab_size, embed_size):
        """ Ctor """
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, x):
        """ Call the embedding function """
        return self.embedding(x)

class positionalEncoder(nn.Module):
    """
        Implements the positional encoder used in Attention is All You Need paper
    """
    def __init__(self, embed_size, max_caption_length=80):
        """
            Ctor
        """
        super(positionalEncoder, self).__init__()
        self.embed_size = embed_size
        # Create a positional encoder "Position and i"
        posEncoder = torch.zeros(max_caption_length, embed_size)
        for pos in range(0, max_caption_length):
            for i in range(max_caption_length):
                posEncoder[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embed_size)))
                posEncoder[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))
                
        # posEncoder is unsqueezed here
        posEncoder = posEncoder.unsqueeze(0)
        # This ensures that the model's parameters aren't trained
        self.register_buffer('posEncoder', posEncoder)
    
    def forward(self, x):
        # Make embeddings larger
        x = x * math.sqrt(self.embed_size)
        seq_len = x.size(1)
        # Store it as a variable without any requirement of a gradient computation.
        if(use_gpu):
            x = x + Variable(self.posEncoder[:,:seq_len], requires_grad=False).cuda()
        else:
            x = x + Variable(self.posEncoder[:,:seq_len], requires_grad=False)
        return x