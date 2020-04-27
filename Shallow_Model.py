import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
class eye_features(nn.Module):
    def __init__(self):
        super(eye_features, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.features.apply(init_weight)

    def forward(self, x):
        x = self.features(x)
        return x

class DeepEyeNet(nn.Module):
    def __init__(self):
        super(DeepEyeNet, self).__init__()
        self.features_L = eye_features()
        self.features_R = eye_features()


        # progress on right eye for spatial weights
        self.spatial_ftrs_L = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.spatial_ftrs_R = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.max_L = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.max_R = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096 + 4096, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(in_features=512, out_features=512),
            # nn.BatchNorm1d(512),
            # nn.Dropout(0.5),
            # nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            # nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=(256 + 2), out_features=2)
        )

        self.spatial_ftrs_L.apply(init_weight)
        self.spatial_ftrs_R.apply(init_weight)
        # self.fc_l.apply(init_weight)
        # self.fc_r.apply(init_weight)
        self.fc1.apply(init_weight)
        self.fc2.apply(init_weight)

    def forward(self, L_eye_input, R_eye_input, headpose_input):
        x_ftrs_L = self.features_L(L_eye_input)
        x_ftrs_R = self.features_R(R_eye_input)

        x_spatial_L = self.spatial_ftrs_L(x_ftrs_L)
        x_spatial_R = self.spatial_ftrs_R(x_ftrs_R)

        x_L = F.dropout(F.relu(torch.mul(x_ftrs_L, x_spatial_L)), 0.5)
        x_R = F.dropout(F.relu(torch.mul(x_ftrs_R, x_spatial_R)), 0.5)

        x_L = self.max_L(x_L)
        x_R = self.max_R(x_R)

        x_L = x_L.view(x_L.size(0), -1)
        x_R = x_R.view(x_R.size(0), -1)

        # x_L = self.fc_l(x_L)
        # x_R = self.fc_r(x_R)

        headpose_input = headpose_input.view(headpose_input.size(0), -1)
        x_cat = torch.cat((x_L, x_R), 1)
        x_fc = self.fc1(x_cat)

        x_fc = torch.cat((x_fc, headpose_input), 1)
        x_fc = self.fc2(x_fc)
        return x_fc

def init_weight(m):
    if type(m) in [nn.Conv2d]:
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, val=0)

    if type(m) in [nn.Linear]:
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, val=0)

    if type(m) in [nn.BatchNorm2d]:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if type(m) in [nn.BatchNorm1d]:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    pass
    model = DeepEyeNet()
    model.cuda()

    summary(model, input_size=[(3, 60, 60), (3, 60, 60), (1, 1, 2)])
