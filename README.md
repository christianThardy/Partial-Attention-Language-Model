# Partial Attention Large Language Model

This ongoing implementation of **PALLM** is a wrapper on top of HuggingFace models to take the strong foundation and simplicity of decoder-only models and enhance them to capture rich contextual representations, like encoder-decoder models. This is made possible through a partial attention mechanism, which selectively attends to different segments of the input text rather than applying a uniform attention across the entire sequence. 

It decreases the effects of attention degeneration as seen in the hallucination problem via a bidirectional attention mask (mechanism to enhance source understanding), a separate positional encoding/specialized language embedding (mechanisms for source-target discrimination) to help the model differentiate between source (prompt) and target (generated output) text sequences. This version of PALLM is different from the original paper as it encourages the model to reconstruct and attend to the input via a source-autoencoder (SAE) objective (mechanism to enforce source grounding). 

By encouraging the model to reconstruct and attend to the input, it stays tethered to the source content during generation. PALLM allows:

  - **Rich Contextual Representation:**
    - Bidirectional encoding of the prompt provides a more thorough understanding of the input.
   
  - **Reduced Hallucinations:**
    - SAE and partial attention help the model stay aligned with the original prompt, minimizing off-topic responses.
   
  - **Long-Form Generation:**
    - Retains the strong autoregressive (decoder-only) capabilities of Hugging Face models for fluent output. 

### Reference:

Fu, Lam, Yu, Cho So, Hu, Liu,  Collier, *Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder*. 2023. [<a href="https://arxiv.org/pdf/2304.04052" rel="nofollow">1</a></li>]
