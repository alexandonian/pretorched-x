import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import pretrainedmodels

pretrained_settings = {
    'trn': {
        'moments': {
            'url': '',
            'num_classes': 339
        }
    }
}


class Relation(torch.nn.Module):
    """Base relation module to model uni-directional relationships.

    A relation maps an ordered set of inputs to a single output representation
    of their uni-directional relationship.

    By convention, the relation is performed on the last two dimensions.

    input[..., num_inputs, in_features] -> output[..., -1, out_features]
    """

    def __init__(self, num_inputs, in_features, out_features, bottleneck_dim=512):
        super().__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck_dim = bottleneck_dim
        self.relate = self.return_mlp()

    def return_mlp(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.num_inputs * self.in_features, self.bottleneck_dim),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.out_features),
        )

    def func(self, input):
        out = self.reshape(input)
        return self.relate(out).view(input.size(0), -1, self.out_features)

    def reshape(self, input):
        return input.contiguous().view(-1, self.num_inputs * self.in_features)

    def forward(self, input):
        """Pass concatenated inputs through simple MLP."""
        return self.func(input)


class MultiScaleRelation(torch.nn.Module):
    """Multi-Relation module.

    This module applies an mlp to that concatenation of
    [2-input relation, 3-input relation, ..., n-input relation].

    """

    def __init__(self,
                 num_input,
                 in_features,
                 out_features,
                 bottleneck_dim=512,
                 num_relations=3):
        super().__init__()
        self.num_input = num_input
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.bottleneck_dim = bottleneck_dim

        # Generate the multiple frame relations
        self.scales = list(range(num_input, 1, -1))
        self.relations_scales = []
        self.subsample_scales = []

        for scale in self.scales:
            relations_scale = self.return_relationset(num_input, scale)
            self.relations_scales.append(relations_scale)
            # Number of inputs in relation to select in each forward pass
            self.subsample_scales.append(
                min(self.num_relations, len(relations_scale)))

        self.relations = nn.ModuleList([
            Relation(scale, self.in_features, self.out_features,
                     self.bottleneck_dim) for scale in self.scales
        ])

        print('Multi-Scale Relation Network Module in use')
        print(['{}-frame relation'.format(i) for i in self.scales])

    def forward(self, input):
        output = []
        for scale in range(len(self.scales)):
            idx_relations = np.random.choice(
                len(self.relations_scales[scale]),
                self.subsample_scales[scale],
                replace=False)
            for idx in idx_relations:
                input_relation = input[..., self.relations_scales[scale][idx], :]
                output.append(self.relations[scale](input_relation))
        return torch.stack(output).sum(0).view(input.size(0), -1, self.out_features)

    def return_relationset(self, num_input, num_input_relation):
        return list(itertools.combinations(range(num_input), num_input_relation))


class HierarchicalRelation(torch.nn.Module):
    """Hierarchical relation module to model nested uni-directional relationships.
    An n-scale hierarchical relation maps an ordered set of inputs to a single
    output representation by recursively computing n-input relations on neighboring
    elements of the output of the previous level.
    """

    def __init__(self, num_inputs, in_features, out_features,
                 relation_size=4, relation_dist=1, bottleneck_dim=1024):
        super().__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features
        self.relation_size = relation_size
        self.relation_dist = relation_dist
        self.bottleneck_dim = bottleneck_dim
        self._prepare_module()

    def _prepare_module(self):
        depth = int(np.ceil((self.num_inputs - self.relation_size) / (self.relation_size - 1)))
        num_inputs_final = self.num_inputs + depth * (1 - self.relation_size)
        self.relations = nn.ModuleList([
            Relation(self.relation_size,
                     self.in_features,
                     self.in_features)
            for _ in range(depth)])
        self.linears = nn.ModuleList([
            nn.Linear(self.in_features,
                      self.out_features)
            for _ in range(depth)])
        self.final_linear = nn.Linear(self.in_features, self.out_features)
        self.final_relation = Relation(num_inputs_final, self.in_features, self.out_features)

    def forward(self, input):
        outs = []
        input = input.view(-1, self.num_inputs, self.in_features)
        for relation, linear in zip(self.relations, self.linears):
            num_inputs = range(input.size(1))
            idx_relations = list(zip(*[num_inputs[i:] for i in range(self.relation_size)]))
            input = torch.stack([relation(input[:, idx, :]) for idx in idx_relations], 1)
            outs.append(linear(input).sum(-2))
        outs.append(self.final_relation(input))
        out = torch.stack(outs).mean(0)
        return out


class MultiScaleHierarchicalRelation(torch.nn.Module):
    """Multi-scale hierarchical relation module."""

    def __init__(self, num_inputs, in_features, out_features,
                 relation_dist=1, bottleneck_dim=512):
        super(MultiScaleHierarchicalRelation, self).__init__()
        self.num_inputs = num_inputs
        self.in_features = in_features
        self.out_features = out_features
        self.relation_dist = relation_dist
        self.bottleneck_dim = bottleneck_dim

        self.scales = range(num_inputs, 1, -1)
        self.num_scales = len(self.scales)
        self.h_relations = nn.ModuleList([
            HierarchicalRelation(num_inputs, in_features, out_features,
                                 relation_size=scale,
                                 relation_dist=relation_dist,
                                 bottleneck_dim=bottleneck_dim)
            for scale in self.scales])
        self.final_relation = Relation(self.num_scales, out_features, out_features,
                                       bottleneck_dim=bottleneck_dim)

    def forward(self, input):
        input = input.contiguous().view(-1, self.num_inputs, self.in_features)
        h_outputs = torch.stack([h_rel(input) for h_rel in self.h_relations], 1)
        h_outputs = h_outputs.view(-1, self.num_scales, self.out_features)
        return self.final_relation(h_outputs)


class TRN(nn.Module):

    def __init__(self, num_classes, num_segments=8, arch='resnet50',
                 frame_bottleneck_dim=1024, video_feature_dim=1024,
                 consensus='HTRN', pretrained='moments',
                 dropout=0.5, partial_bn=True):
        super().__init__()
        self.arch = arch
        self.reshape = True
        self.dropout = dropout
        self._enable_pbn = True
        self.consensus = consensus
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.video_feature_dim = video_feature_dim
        self.frame_bottleneck_dim = frame_bottleneck_dim

        num_pc = 1000 if pretrained == 'imagenet' else 339
        self.base_model = pretrainedmodels.__dict__[arch](num_pc, pretrained)
        self.frame_feature_dim = self.base_model.last_linear.in_features
        self.base_model.last_linear = torch.nn.Dropout(self.dropout)
        self.std = self.base_model.std
        self.mean = self.base_model.mean
        self.input_size = self.base_model.input_size[1:]
        self.input_space = self.base_model.input_space

        consensus_mods = {
            'TRN': Relation,
            'HTRN': HierarchicalRelation,
            'MSTRN': MultiScaleRelation,
            'MSHTRN': MultiScaleHierarchicalRelation,
        }

        try:
            temporal_relation = consensus_mods[consensus]
        except KeyError:
            raise ValueError('Unrecognized temporal consensus.')
        else:
            self.temporal_relation = temporal_relation(self.num_segments,
                                                       self.frame_feature_dim,
                                                       self.video_feature_dim,
                                                       self.frame_bottleneck_dim)
            self.last_linear = nn.Linear(self.video_feature_dim, self.num_classes)

        print(('Initializing {} with base arch: {}.\n'
               'Configuration:\n\t'
               'num_segments:          {}\n\t'
               'frame_feature_dim:     {}\n\t'
               'frame_bottleneck_dim:  {}\n\t'
               'temporal_consensus:    {}\n'
               .format(self.__class__.__name__, self.arch,
                       self.num_segments, self.frame_feature_dim,
                       self.frame_bottleneck_dim, self.consensus)))

    def features(self, input):
        # Feature representations from base model
        batch_size = input.size(0)
        base_rep = self.base_model(input.view((-1, 3) + input.size()[-2:]))
        base_rep = base_rep.view(batch_size, -1, self.num_segments, base_rep.size(-1))
        num_inputs = base_rep.size(1)

        # Relation over video frames for each input
        t_in = base_rep.view(-1, num_inputs, self.num_segments, base_rep.size(-1))
        return self.temporal_relation(t_in).squeeze()

    def logits(self, features):
        return self.last_linear(features)

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super().train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0, 'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0, 'name': "BN scale/shift"},
        ]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size[0] * 256 // 224

    # def get_augmentation(self):
        # return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
        #    GroupRandomHorizontalFlip(is_flow=False)])


def trn(num_classes=339, num_segments=8, consensus='MSTRN', arch='resnet50',
        pretrained='moments', frame_bottleneck_dim=1024, video_feature_dim=1024):
    if pretrained:
        settings = pretrained_settings['trn'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        model = TRN(num_classes=num_classes, num_segments=num_segments, arch=arch)
        model.load_state_dict(model_zoo.load_url(settings['url']))
    else:
        model = TRN(num_classes=num_classes, num_segments=num_segments, arch=arch)
    return model


if __name__ == '__main__':

    batch_size = 1
    num_frames = 8
    num_classes = 174
    img_feature_dim = 512
    input_var = torch.autograd.Variable(torch.randn(batch_size, num_frames, 3, 224, 224))
    model = trn(num_classes=10, pretrained=None)
    output = model(input_var)
    # print(output)
    features = model.features(input_var)
    # print(features)
    logits = model.logits(features)
    # print(logits)

    assert trn(num_classes=10, pretrained=None)
    print('success')
    assert trn(num_classes=339, pretrained='moments')
    print('success')
