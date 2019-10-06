from src.model.nets.backbone import resnet3d

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet3d':
        return resnet3d.ResNet50(output_stride, BatchNorm)
    else:
        raise NotImplementedError
