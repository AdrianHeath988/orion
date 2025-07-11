import torch.nn as nn
import orion.nn as on

class LoLA(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 5, kernel_size=2, padding=0, stride=2)
        self.bn1 = on.BatchNorm2d(5)
        self.act1 = on.Quad()
        
        # self.fc1 = on.Linear(980, 100)
        self.fc1 = on.Linear(980, 32)
        # self.bn2 = on.BatchNorm1d(100)
        self.bn2 = on.BatchNorm1d(32)
        self.act2 = on.Quad()
        
        # self.fc2 = on.Linear(100, num_classes)
        self.fc2 = on.Linear(32, num_classes)
        self.flatten = on.Flatten()

    def forward(self, x): 
        print("[DEBUG] forward act 1")
        x = self.act1(self.bn1(self.conv1(x)))
        print("[DEBUG] forward flatten")
        x = self.flatten(x)
        print("[DEBUG] forward act 2")
        x = self.act2(self.bn2(self.fc1(x)))
        print("[DEBUG] forward finished")
        return self.fc2(x)
 

if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = LoLA()
    net.eval()

    x = torch.randn(1,1,28,28)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (1,28,28), depth=10)
    print("Total flops: ", total_flops)
