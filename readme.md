# Generating faces from visual attributes

Project Objectives: The work focuses on generating realistic images of human faces using computational methods. The main goal is the development and training of a network capable of generating portrait images of individuals, conditioned by various attributes. These attributes start with the simplest, gender (female or male), and then include other characteristics such as smile, hair color, age, and the presence of glasses. To achieve this goal, a conditioned generative adversarial network model has been implemented, consisting of two main components: the generator and the discriminator. The generator is a convolutional network that takes as input a noise vector and the desired attributes, processes them through convolutional layers, and generates an image. The discriminator, the second network, analyzes the generated images to determine whether they are real or not. Based on the discriminator's outputs, a cost function is calculated and used for training the model.

![image](https://github.com/teodora-vaman/Face_Synthesis/assets/86794414/b657cee9-69f7-483e-9f1d-f0db03ff8fb3)

![image](https://github.com/teodora-vaman/Face_Synthesis/assets/86794414/e104e1d1-2752-496f-8543-1fcd1c17b9bd)

However, the results generated by this model exhibit artifacts and noise, requiring a more advanced approach to solve the problem. To overcome these difficulties, the work chooses to implement a scientific article that proposes a remarkable method for synthesizing human faces based on attributes, using VAE (variational autoencoder) and GAN (generative adversarial network) networks. The selected article divides the problem into three distinct stages, each aimed at generating a more realistic face. The first stage, based on a conditioned variational autoencoder, takes the attributes and noise vector and attempts to outline a rudimentary face that meets the imposed requirements. The second stage takes the rudimentary image and learns to refine the details, resulting in a black-and-white image that can represent a pencil sketch of a human face. In the third stage, a GAN network receives the result of the previous stage and learns to add colors and other necessary details to assert that the resulting person is real.

In total, we analyzed and tested four systems capable of generating human faces, referred to in the paper as CGAN-64, CGAN-128, A2F-noise, and A2F-sketch. The generated images were evaluated subjectively and objectively using metrics such as SSIM score, Inception score, and Fréchet distance. The conclusions obtained from the experiments are as follows:

The CGAN-64 model produces images with people, but they exhibit a high level of artifacts. An advantage of the network is the large diversity of the generated images. By increasing the resolution from 64x64 to 128x128, the images generated by CGAN-128 have significantly improved, becoming clearer and more realistic. However, the network does not produce diverse results. The A2F-noise model combines state-of-the-art features and presents clear and diverse results, as presented in the reference article. However, in practice, the network got stuck in generating the same results, similar to CGAN-128. Even though the images are realistic and conditioned by various attributes, we cannot assert that the system is a high-performing generator. The A2F-sketch model is identical to the one presented earlier, with the difference being the input, which in this case is the outlined sketch of a portrait. The obtained results exhibit artifacts, but the colors are realistic, conditioning based on attributes produces different features, and the generated images exhibit a higher degree of diversity. However, some instances were observed where the model generated random or unexpected results. This aspect can be attributed to the complexity of the image generation process and the interaction between the different components of the model.

![image](https://github.com/teodora-vaman/Face_Synthesis/assets/86794414/b0724bb1-94f0-44d2-8424-7909bf1f45ee)

![image](https://github.com/teodora-vaman/Face_Synthesis/assets/86794414/9ef9de3f-3f30-4160-801a-ab322e44b86a)


# References
Di, Xing, and Vishal M. Patel. "Face synthesis from visual attributes via sketch using conditional vaes and gans." _arXiv preprint arXiv:1801.00077_ (2017).
```
@article{di2017face,
  title={Face synthesis from visual attributes via sketch using conditional vaes and gans},
  author={Di, Xing and Patel, Vishal M},
  journal={arXiv preprint arXiv:1801.00077},
  year={2017}
}
```
