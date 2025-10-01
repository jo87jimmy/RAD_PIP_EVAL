import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Helper Blocks (Reconstructive SubNetwork)
# ==============================================================================
class EncoderReconstructive(nn.Module):

    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.out_channels = [
            base_width, base_width * 2, base_width * 4, base_width * 8,
            base_width * 8
        ]

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return [b1, b2, b3, b4, b5]


class DecoderReconstructive(nn.Module):

    def __init__(self,
                 base_width,
                 out_channels=3
                 ):  # Adjusted default out_channels for reconstruction
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 1, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 1),
            nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
            # No activation here, reconstruction is often linear.
            # If input is [0,1], output might need to be clamped or scaled later.
        )

    def forward(self, b5):
        up1 = self.up1(b5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out


class ReconstructiveSubNetwork(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        # Ensure out_channels is passed correctly
        self.decoder = DecoderReconstructive(base_width,
                                             out_channels=out_channels)

    def forward(self, x):
        features = self.encoder(x)
        # 解碼器只使用最後一層特徵來重建圖像
        output = self.decoder(features[-1])
        return output


# ==============================================================================
# Helper Blocks (Discriminative SubNetwork)
# ==============================================================================
class EncoderDiscriminative(nn.Module):

    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Module):
    # out_channels should be 1 for anomaly map (binary segmentation)
    def __init__(self, base_width, out_channels=1):  # Corrected default to 1
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8),
                      base_width * 8,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8),
                      base_width * 4,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4),
                      base_width * 2,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1),
                      base_width,
                      kernel_size=3,
                      padding=1), nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True))

        # 修正這裡：將 fin_out 變回 nn.Sequential
        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b, b5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, b4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, b1), dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out


class DiscriminativeSubNetwork(nn.Module):

    def __init__(self,
                 in_channels=6,
                 out_channels=1,
                 base_channels=64):  # out_channels corrected to 1
        super(DiscriminativeSubNetwork, self).__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels,
                                                     base_channels)
        self.decoder_segment = DecoderDiscriminative(
            base_channels,
            out_channels=out_channels)  # Pass corrected out_channels
        self.sigmoid = nn.Sigmoid()  # Add sigmoid for probability output

    def forward(self, x, return_feats=False):
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        # Raw logits from the decoder
        raw_seg_map = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        # Apply sigmoid to get probabilities for the anomaly map
        seg_map = self.sigmoid(raw_seg_map)

        if return_feats:
            features = [b1, b2, b3, b4, b5, b6]
            return seg_map, features
        else:
            return seg_map


# ==============================================================================
# Main Anomaly Detection Model
# ==============================================================================
class AnomalyDetectionModel(nn.Module):

    def __init__(self, recon_in, recon_out, recon_base, disc_in, disc_out,
                 disc_base):
        super(AnomalyDetectionModel, self).__init__()
        self.reconstruction_subnet = ReconstructiveSubNetwork(
            in_channels=recon_in,
            out_channels=recon_out,
            base_width=recon_base)
        self.discriminator_subnet = DiscriminativeSubNetwork(
            in_channels=disc_in,
            out_channels=
            disc_out,  # This should now be 1 due to the change in DiscriminativeSubNetwork
            base_channels=disc_base)

    def forward(self, x, return_feats=False):
        recon_image = self.reconstruction_subnet(x)
        disc_input = torch.cat((x, recon_image), dim=1)

        if return_feats:
            seg_map, features = self.discriminator_subnet(disc_input,
                                                          return_feats=True)
            return recon_image, seg_map, features
        else:
            seg_map = self.discriminator_subnet(disc_input, return_feats=False)
            return recon_image, seg_map
