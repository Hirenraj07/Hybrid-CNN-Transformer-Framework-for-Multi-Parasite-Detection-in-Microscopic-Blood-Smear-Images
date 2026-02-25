import tensorflow as tf

# Load architecture
model = tf.keras.models.load_model("hybrid_efficientnet_vit_finetuned_99.keras")

# Load weights
model.load_weights("hybrid_weights_finetuned.h5")
