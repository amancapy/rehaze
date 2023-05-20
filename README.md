# rehaze
Diffusion-like Iterative Image Dehazing with Deep Learning

This project was inspired by [Cold Diffusion](https://arxiv.org/abs/2208.09392), although this isn't a generative model.

Using the distance-based scattering model $x_{t} = x_{0} e^{-\beta d} + \alpha (1 - e^{-\beta d})$ you can very closely approximate $x_{\frac{t}{2}}$ given $x_{t}$ and $\hat{x_{0}}$<sup>\*</sup>, which means that you can iteratively re-haze and dehaze your hazy image until satisfied *without knowing depth information during inference*, using $x_{t}$ at every iterative step staying loyal to the input. You get the best of both worlds -- the model's learned intuition of the color distributions over clear scenes, and the true physical structure from the $x_{0}$ input. 

The [NYU-DepthV2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) was very helpful, with its high-quality depth maps. I struggled to find large depth-pair datasets for *outdoor* scenes so I had to make do with indoor scenes.

I started this project originally as an implementation of the Diffusion model, and you will find almost all of the diffusion code fully intact, commented out.

| ![image](https://github.com/amancapy/rehaze/assets/111729660/4721a214-4c81-450d-b024-1bf64844f4df) |
|:--:|
| Hazy input -> Ground Truth -> Prediction |
| The first  $`\hat{x_{0}}`$ prediction made after only 2000 small batches of training.
Note that 1) the input was chosen intentionally to be extremely hazy. 2) this was only a 6M-parameter model trained on my personal laptop and of course a larger model will require very few iterations to make "perfect" final results.|

\*<sub>I will eventually get around to LaTeXing my formulation</sub>
