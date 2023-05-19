# rehaze
diffusion-like iterative image dehazing

This project was inspired by [Cold Diffusion](https://arxiv.org/abs/2208.09392), although this isn't a generative model.

Using the distance-based scattering model $x_{t} = x_{0} e^{-\beta d} + \alpha (1 - e^{-\beta d})$ you can very closely approximate $x_{\frac{t}{2}}$ given $x_{t}$ and $\hat{x_{0}}$, which means that you can iteratively re-haze and dehaze your hazy image until satisfied, using $x_{t}$ at every iterative step staying loyal to the input. You get the best of both worlds -- the model's learned intuition of the color distributions over clear scenes, and the true physical structure from the $x_{0}$ input. 

I started this project originally as an implementation of the Diffusion model, and you will find almost all of the diffusion code fully intact, commented out.

| ![image](https://github.com/amancapy/rehaze/assets/111729660/4721a214-4c81-450d-b024-1bf64844f4df) |
|:--:|
| A single-shot prediction made after only 2000 small batches of training. Note that this is not an iteratively taken sample, but the first $`\hat{x_{0}}`$ prediction. Of course iterative sampling will give an *even* better result. |
