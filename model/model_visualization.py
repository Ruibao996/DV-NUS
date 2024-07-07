import tensorflow as tf
from tensorflow.keras.utils import plot_model

def visualize_model(model_path, output_image_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Visualize the model
    plot_model(model, to_file=output_image_path, show_shapes=True, show_layer_names=True)
    print(f'Model visualization saved to {output_image_path}')

if __name__ == "__main__":
    model_path = './../result/lstm_model.h5'  # Path to the saved model
    output_image_path = './../result/lstm_model_visualization.png'  # Path to save the model visualization

    visualize_model(model_path, output_image_path)
