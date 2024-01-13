**Score: 9.5 / 10**

* Good code-review summary.
  - The reason it's working without the -2 is because you have `predictions .- targets`

## Jupyter notebook demonstration
* The `scratchwork` notebook achieves some of this, but the instructions were to prepare a cleaned-up demonstration.
  - Rename the notebook to something other than `scratchwork`.
  - Remove all the extra cells at the bottom testing various things out.
  - Put a bit more effort into how it's presented.

## single_neuron_training.jl
* line 35 should be `targets .- predictions`, but this cancels out with dropping the minus sign on lines 39 & 42.
* line 74: if you actually used this jumble of weights and biases, it would be hard to interpret.
