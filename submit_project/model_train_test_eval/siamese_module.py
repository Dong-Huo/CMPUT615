import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SiameseNetwork(nn.Module):
    def __init__(self, backbone, num_mtx_params=8):
        super(SiameseNetwork, self).__init__()

        self.features = nn.Sequential(*list(backbone.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_mtx_params),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        cat_output = torch.cat((output1, output2), dim=1)
        cat_output = self.classifier(cat_output)

        return cat_output

class MSELoss(torch.nn.Module):

    def __init__(self,l_batch_size, reg_coord, warp_sty, stage='train'):
        super(MSELoss, self).__init__()
        # self.batchSize = l_batch_size
        self.warp_sty = warp_sty
        self.reg_coord = reg_coord
        self.stage = stage

    def forward(self, cat_output, label):
        # euclidean_distance = F.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        target = cat_output
        batch_size = cat_output.size()[0]
        O = torch.zeros(batch_size, dtype=torch.float32).cuda()
        I = torch.ones(batch_size, dtype=torch.float32).cuda()

        if self.reg_coord:

            if self.warp_sty == "homo":
                p1,p2,p3,p4,p5,p6,p7,p8 = torch.unbind(cat_output,dim=1)
                pMtx = torch.stack([torch.stack([p1,p2,p3],dim=-1),
                                    torch.stack([p4,p5,p6],dim=-1),
                                    torch.stack([p7,p8,I] ,dim=-1)], dim=1)
            if self.warp_sty == "affine":
                p1, p2, p3, p4, p5, p6= torch.unbind(cat_output, dim=1)
                pMtx = torch.stack([torch.stack([p1, p2, p3], dim=-1),
                                    torch.stack([p4, p5, p6], dim=-1),
                                    torch.stack([O, O, I], dim=-1)], dim=1)
            if self.warp_sty == "simi":
                pc, ps, tx,ty = torch.unbind(cat_output, dim=1)
                pMtx = torch.stack([torch.stack([pc, -ps, tx], dim=-1),
                                    torch.stack([ps, pc, ty], dim=-1),
                                    torch.stack([O, O, I], dim=-1)], dim=1)


            refMtrx = torch.from_numpy(np.eye(3).astype(np.float32)).cuda()
            refMtrx = refMtrx.repeat(batch_size,1,1)
            transMtrx = refMtrx.matmul(pMtx)

            X,Y = np.meshgrid(np.linspace(0,223,2),np.linspace(0,223,2))
            X,Y = X.flatten(), Y.flatten()
            XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
            XYhom = np.tile(XYhom, [batch_size,1,1]).astype(np.float32)
            XYhom = torch.from_numpy(XYhom).cuda()

            XYwarpHom = transMtrx.matmul(XYhom)
            XwarpHom, YwarpHom, ZwarpHom = torch.unbind(XYwarpHom,dim=1)
            # Xwarp = (XwarpHom / (ZwarpHom + 1e-8)).reshape(self.batchSize, 2, 2)
            # Ywarp = (YwarpHom / (ZwarpHom + 1e-8)).reshape(self.batchSize, 2, 2)

            Xwarp = (XwarpHom / (ZwarpHom + 1e-8))
            Ywarp = (YwarpHom / (ZwarpHom + 1e-8))

            target = torch.cat([Xwarp, Ywarp], dim=1)

        if self.stage == 'train':
            l2_loss = F.mse_loss(target, label)
            return l2_loss
        elif self.stage == 'test':
            return target
