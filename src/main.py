from models import VanillaConvAE, BetaConvVAE
from data_generator import DataGenerator

if __name__ == "__main__":

    #Load Data
    data_generator = DataGenerator(folder_name="data/dSprites/",
                                   image_size=100,
                                   data_split=0.8, 
                                   number_to_augment=0)

    x_train, x_valid, x_test = data_generator.generate()

    # cherry_picked_data = data_generator.cherry_pick(folder_name="cherry_picked")

    # Vanilla Convolutional Autoencoder
    # vanilla_conv_ae = VanillaConvAE(data_generator=data_generator)
    # vanilla_conv_ae.build()
    # vanilla_conv_ae.compile(optimizer="adadelta", loss="binary_crossentropy")
    # vanilla_conv_ae.fit(x=x_train, 
    #             y=x_train,
    #             epochs=150,
    #             batch_size=128,
    #             shuffle=True,
    #             validation_data=(x_valid, x_valid))

    # encoded_imgs = vanilla_conv_ae.encode(x_test)
    # decoded_imgs = vanilla_conv_ae.predict(x_test)

    # vanilla_conv_ae.plot_learning_curve()
    # vanilla_conv_ae.reconstruct(encoded_imgs, decoded_imgs, x_test)
    # vanilla_conv_ae.inspect_latent(encoded_imgs)

    # Beta Variational Convolutional Autoencoder
    beta_conv_vae = BetaConvVAE(beta=20, data_generator=data_generator)
    beta_conv_vae.build()
    beta_conv_vae.compile(optimizer="adadelta")
    beta_conv_vae.fit(x=x_train, 
                y=None,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_valid, None))

    encoded_imgs = beta_conv_vae.encode(x_test)
    decoded_imgs = beta_conv_vae.predict(x_test)

    # print(encoded_imgs)
    beta_conv_vae.plot_learning_curve()
    beta_conv_vae.plot_real_vs_reconstruct(decoded_imgs, x_test)
    beta_conv_vae.plot_latent_vectors(encoded_imgs)
    beta_conv_vae.plot_sampled_images()

    # beta_conv_vae.plot_latent_distribution(cherry_picked_data)

    # Save Model
    beta_conv_vae.save_model()

    beta_conv_vae_1 = BetaConvVAE(beta=20, data_generator=data_generator)
    beta_conv_vae_1.load_model()

    beta_conv_vae_1.plot_latent_distribution(cherry_picked_data)

    # Variational Convolutional Autoencoder with Auxiliary Cost Functions
    