import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np


def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def load_model(weights_path, device):
    model = models.densenet121()
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    return model


class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
        self.gradients = None

    def hook_fn(self, module, input, output):
        self.features = output
        output.register_hook(self.save_grads)

    def save_grads(self, grads):
        self.gradients = grads

    def remove(self):
        self.hook.remove()


def generate_gradcam(model, input_tensor, device, target_layer):
    model.eval()

    # Hook the feature and gradient for the target layer
    features_hook = SaveFeatures(target_layer)

    # Forward pass
    output = model(input_tensor.to(device))
    target_class = output.argmax(dim=1).item()

    # Backward pass to get the gradients
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    # Get the feature maps and gradients
    gradients = features_hook.gradients[0].cpu().data.numpy()
    features = features_hook.features[0].cpu().data.numpy()

    # Compute the Grad-CAM
    weights = np.mean(gradients, axis=(1, 2))
    gradcam_map = np.sum(weights[:, np.newaxis, np.newaxis] * features, axis=0)
    gradcam_map = np.maximum(gradcam_map, 0)
    gradcam_map = cv2.resize(
        gradcam_map, (input_tensor.shape[3], input_tensor.shape[2]))
    gradcam_map = gradcam_map - np.min(gradcam_map)
    gradcam_map = gradcam_map / np.max(gradcam_map)

    # Visualize the Grad-CAM output
    img = input_tensor.cpu().numpy()[0].transpose((1, 2, 0))
    img = np.float32(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)

    # Adjust transparency
    alpha = 0.01  # Make the heatmap more transparent
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = superimposed_img / np.max(superimposed_img)

    return superimposed_img
