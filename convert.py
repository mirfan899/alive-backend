import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# Load the ResNet50 model with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add global average pooling
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

# Create the modified model
model = Model(inputs=base_model.input, outputs=x)

# Print Keras model summary to verify input/output shapes
print("Keras Model Summary:")
model.summary()

# Convert to TFLite with explicit input shape
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

# Explicitly set the input shape
converter.experimental_new_converter = True  # Use new converter to avoid legacy issues
converter.representative_dataset = None  # No quantization dataset
tflite_model = converter.convert()

# Save the TFLite model
with open("resnet50_imagenet.tflite", "wb") as f:
    f.write(tflite_model)

# Verify TFLite model input/output
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("\nTFLite Input Details:", input_details)
print("TFLite Output Details:", output_details)

# Test inference with dummy input
import numpy as np
dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output Shape from Dummy Inference:", output_data.shape)