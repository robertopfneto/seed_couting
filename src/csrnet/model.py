import torch
import torch.nn as nn
import torchvision.models as models

# Congested Scene Recognition Network:
"""
3 principais blocos:
- Frontend: extrai recursos visuais com uma VGG16 pré-treinada
- Backend: aplica convoluções dilatadas para expandir o campo de visão (receptive field) sem perder resolução
- Output Layer: gera um Density Map, onde cada valor dos pixels indica presença de sementes
"""

class CSRNet(nn.Module):
    def __init__(self, load_pretrained=True):
        super(CSRNet, self).__init__() 

        # VGG16 até a camada conv4_3
        vgg = models.vgg16_bn(pretrained=load_pretrained)
        features = list(vgg.features.children())[:33]
        self.frontend = nn.Sequential(*features)

        # convoluções dilatadas (sem perder resolução)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        """ 
            dilation: aumenta virtualmente o tamanho do kernel para um campo de visao 5x5
            padding: evita que a imagem fique menor após a convolução
        """

        # 🔷 OUTPUT: mapa de densidade (1 canal)
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
