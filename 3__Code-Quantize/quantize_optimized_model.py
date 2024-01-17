import torch
import torch.quantization

MDL_PATH = '/home/mwilkers1/Documents/Projects/IMPACT-EdgeAI-Mars/4__Saved_ResNet50-MSL-v2.1/99percent-ResNet50-v2.1/mars_classifier_scripted.pt'

# Load your model
model_fp32 = torch.jit.load(MDL_PATH)
model_fp32.eval()  # Make sure it's in evaluation mode

# Set qconfig for the model
qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32.qconfig = qconfig

# Apply the qconfig to all submodules
torch.quantization.prepare(model_fp32, inplace=True)

# Continue with the quantization process
model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace=False)
model_fp32_quantized = torch.quantization.convert(model_fp32_prepared, inplace=False)