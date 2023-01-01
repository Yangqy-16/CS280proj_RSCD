from models.ChangeFormer import *
from models.SNUNet_CD import *


class DecoderTransformer_Ours(nn.Module):
    """ Transformer Decoder """
    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True, 
                    in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2, 
                    decoder_softmax=False, feature_strides=[2, 4, 8, 16]):
        super(DecoderTransformer_Ours, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index        = in_index
        self.align_corners   = align_corners
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)  # 256-->64
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)  # 128-->64
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)  # 64-->64
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)  # 32-->64

        # convolutional Difference Modules
        self.diff_c4   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)  # 128-->64
        self.diff_c3   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)  # 128-->64
        self.diff_c2   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)  # 128-->64
        self.diff_c1   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)  # 128-->64

        # taking outputs from middle of the encoder
        self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)  # 64-->2
        self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)  # 64-->2
        self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)  # 64-->2
        self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)  # 64-->2

        # ECAM
        self.ca = ChannelAttention(self.embedding_dim * 4, ratio=self.embedding_dim//2)
        self.ca1 = ChannelAttention(self.embedding_dim, ratio=self.embedding_dim//8)

        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim*len(in_channels), out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )  # 64*4-->64

        # Final prediction head
        self.convd2x    = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2, padding=1)
        self.dense_2x   = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x    = nn.ConvTranspose2d(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2, padding=1)
        self.dense_1x   = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = nn.Conv2d(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        
        # Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid() 

    def _transform_inputs(self, inputs):
        """ Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs.
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':  # this is chosen
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):
        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, [1/2, 1/4, 1/8, 1/16]
        x_2 = self._transform_inputs(inputs2)  # len=4, [1/2, 1/4, 1/8, 1/16]

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])  # (n, 64, h4, w4)
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])  # (n, 64, h4, w4)
        _c4   = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))  # (n, 128, h4, w4)
        p_c4  = self.make_pred_c4(_c4)  # (n, 2, h4, w4)
        outputs.append(p_c4)
        _c4_up= resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])  # (n, 64, h3, w3)
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])  # (n, 64, h3, w3)
        _c3   = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        p_c3  = self.make_pred_c3(_c3)
        outputs.append(p_c3)
        _c3_up= resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        _c2   = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        p_c2  = self.make_pred_c2(_c2)
        outputs.append(p_c2)
        _c2_up= resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        _c1   = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        p_c1  = self.make_pred_c1(_c1)
        outputs.append(p_c1)

        # ECAM
        out = torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1)  # F_ensemble        
        intra = torch.sum(torch.stack((_c4_up, _c3_up, _c2_up, _c1)), dim=0)
        ca1 = self.ca1(intra)  # ca1 = M_intra
        out = self.ca(out) * (out + ca1.repeat(1, 4, 1, 1))  # ECAM

        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(out)  # (n, 64, h1, w1)

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)  # (n, 64, 2*h1, 2*w1)
        # Residual block
        x = self.dense_2x(x)  # (n, 64, 2*h1, 2*w1)
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)  # (n, 64, 4*h1, 4*w1)
        # Residual block
        x = self.dense_1x(x)  # (n, 64, 4*h1, 4*w1)

        # Final prediction
        cp = self.change_probability(x)  # (n, 2, H, W)
        
        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class OurNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=256):
        super(OurNet, self).__init__()
        
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths     = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1 

        self.Tenc_x2 = EncoderTransformer_v3(img_size=256, patch_size=7, in_chans=input_nc, num_classes=output_nc, embed_dims=self.embed_dims,
                 num_heads = [1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=self.drop_rate,
                 attn_drop_rate = self.attn_drop, drop_path_rate=self.drop_path_rate, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=self.depths, sr_ratios=[8, 4, 2, 1])
        
        # Transformer Decoder
        self.TDec_x2 = DecoderTransformer_Ours(input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=False, 
                    in_channels=self.embed_dims, embedding_dim=self.embedding_dim, output_nc=output_nc, 
                    decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16])

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        cp = self.TDec_x2(fx1, fx2)
        return cp

