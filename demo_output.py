import torch
import torch.nn as nn

# Define Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mean = nn.Parameter(torch.randn(out_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 5)

    def forward(self, x, sample=True):
        if sample:
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mean + bias_std * torch.randn_like(bias_std)
        else:
            weight = self.weight_mean
            bias = self.bias_mean
        return torch.nn.functional.linear(x, weight, bias)

# Define Bayesian Head
class BayesianDetectionHead(nn.Module):
    def __init__(self, input_channels=512):
        super().__init__()
        self.fc1 = BayesianLinear(input_channels, 256)
        self.fc2 = BayesianLinear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, sample=True):
        x = self.fc1(x, sample)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x, sample)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# Define B_YOLO Model
class B_YOLO(nn.Module):
    def __init__(self, backbone, num_classes=80):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.bayesian_head = BayesianDetectionHead(512)
        self.detection_head = nn.Linear(128, 4 + num_classes)
        self.classification_head = nn.Linear(128, num_classes)
        self.pose_head = nn.Linear(128, 17 * 2)
        self.segmentation_head = nn.Linear(128, 256)
        self.oriented_head = nn.Linear(128, 5 + num_classes)
        self.current_task = None

    def set_task(self, task):
        self.current_task = task

    def forward(self, x, sample=True, num_mc_samples=1, return_uncertainty=True):
        with torch.no_grad():
            features = self.backbone(x)
        if features.dim() == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        
        if return_uncertainty and num_mc_samples > 1:
            predictions_list = []
            for _ in range(num_mc_samples):
                z = self.bayesian_head(features, sample=sample)
                if self.current_task == 'classification':
                    pred = self.classification_head(z)
                elif self.current_task == 'detection':
                    pred = self.detection_head(z)
                elif self.current_task == 'pose':
                    pred = self.pose_head(z)
                predictions_list.append(pred)
            predictions_stacked = torch.stack(predictions_list, dim=0)
            mean_predictions = predictions_stacked.mean(dim=0)
            uncertainty = predictions_stacked.var(dim=0)
            return {'predictions': mean_predictions, 'uncertainty': uncertainty, 'num_mc_samples': num_mc_samples}
        else:
            x = self.bayesian_head(features, sample=sample)
            if self.current_task == 'classification':
                predictions = self.classification_head(x)
            elif self.current_task == 'detection':
                predictions = self.detection_head(x)
            elif self.current_task == 'pose':
                predictions = self.pose_head(x)
            return {'predictions': predictions, 'uncertainty': torch.zeros_like(predictions), 'num_mc_samples': 1}

# Create dummy backbone
class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 512, 1)
    def forward(self, x):
        return self.conv(x)

# Initialize model and create dummy input
print('='*80)
print('MODEL INPUT/OUTPUT DEMONSTRATION')
print('='*80)

backbone = DummyBackbone()
model = B_YOLO(backbone, num_classes=5)
model.eval()

# Create dummy input batch
dummy_input = torch.randn(2, 3, 224, 224)
print('\n✓ DUMMY INPUT CREATED')
print(f'  Input shape: {dummy_input.shape}')
print(f'  (2 images, 3 channels, 224x224 pixels)')

# CLASSIFICATION TASK
print('\n' + '-'*80)
print('TEST 1: CLASSIFICATION OUTPUT')
print('-'*80)
model.set_task('classification')
with torch.no_grad():
    output = model(dummy_input, num_mc_samples=1, return_uncertainty=False)
print(f'\n✓ Raw predictions shape: {output["predictions"].shape}')
print(f'  Image 1 raw logits: {output["predictions"][0].numpy()}')
print(f'  Image 2 raw logits: {output["predictions"][1].numpy()}')

probs = torch.softmax(output['predictions'], dim=1)
print(f'\n✓ After softmax (probabilities):')
print(f'  Image 1: {probs[0].numpy()}')
print(f'  Image 2: {probs[1].numpy()}')

# WITH UNCERTAINTY
print('\n' + '-'*80)
print('TEST 2: CLASSIFICATION WITH UNCERTAINTY (5 MC samples)')
print('-'*80)
with torch.no_grad():
    output = model(dummy_input, num_mc_samples=5, return_uncertainty=True)
print(f'\n✓ Mean predictions: {output["predictions"][0].numpy()}')
print(f'  Uncertainty: {output["uncertainty"][0].numpy()}')

# DETECTION TASK
print('\n' + '-'*80)
print('TEST 3: DETECTION OUTPUT')
print('-'*80)
model.set_task('detection')
with torch.no_grad():
    output = model(dummy_input, num_mc_samples=3, return_uncertainty=True)
print(f'\n✓ Detection output shape: {output["predictions"].shape}')
print(f'  BBox [x1,y1,x2,y2]: {output["predictions"][0, :4].numpy()}')
print(f'  Class logits: {output["predictions"][0, 4:].numpy()}')
print(f'\n  Uncertainty in BBox: {output["uncertainty"][0, :4].numpy()}')
print(f'  Uncertainty in Classes: {output["uncertainty"][0, 4:].numpy()}')

print('\n' + '='*80)
print('✓ SUCCESS - MODEL OUTPUTS DISPLAYED!')
print('='*80)
