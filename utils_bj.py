for n, param in self.models["encoder"].stages.named_parameters():
    if 'BlockGatingUnit' in n and 'Dense' in n:
        if 'weight' in n:
            nn.init.zeros_(param)
        if 'bias' in n:
            nn.init.ones_(param)
        print(n, param)

for n, param in self.models["encoder"].stages.named_parameters():
    if 'BlockGatingUnit' in n and 'Dense' in n:
        print(n, param)