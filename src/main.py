from models import VanillaConvAE
from data_generator import DataGenerator

if __name__ == "__main__":

    #Load Data
    data_generator = DataGenerator(folder_name="data/etg/",
                                   number_of_images=150,
                                   image_size=100,
                                   data_split=0.8)

    x_train, x_valid, x_test = data_generator.generate()

    # Vanilla Convolutional Autoencoder
    vanilla_conv_ae = VanillaConvAE()
    vanilla_conv_ae.build()
    vanilla_conv_ae.compile(optimizer="adadelta", loss="binary_crossentropy")
    vanilla_conv_ae.fit(x=x_train, 
                y=x_train,
                epochs=150,
                batch_size=128,
                shuffle=True,
                validation_data=(x_valid, x_valid))

    encoded_imgs = vanilla_conv_ae.encode(x_test)
    decoded_imgs = vanilla_conv_ae.predict(x_test)

    vanilla_conv_ae.plot_learning_curve()
    vanilla_conv_ae.inspect_result(encoded_imgs, decoded_imgs, x_test)

    # Variational 2D Convolutional Autoencoder

    # Beta Variational Convolutional Autoencoder

    # Variational Convolutional Autoencoder with Auxiliary Cost Functions
    