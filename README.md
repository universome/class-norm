Our setup:
- Our main model is discriminator, which performs classification. The setup is very similar to CIZSL paper in that sense.
- Next, we have an additional A-GEM loss, which makes discriminator not to forget older examples.

Ideas:
- We can use class prototypes as task descriptors. To get a prototype we can just compute average VGG feature. Of course, this will be cheating, but will be a good PoC.
- Like in editable neural networks, but we can make outputs to be unchanged for just some random inputs.
- Can we change episodic memory in the following way. We train a classification model together with a generative one. And instead of storing examples, we just generate them. We can generate them with the corresponding class labels. And we can do it without class labels by the following way. On each iteration we generate a batch of examples, remember our model's outputs on them. And then perform an optimization step in such a way that outputs for a generated batch does not change (either by regularization or by projecting the gradient).
- Is it really better to use episodic memory for projecting the gradient? Maybe it will be more beneficial just to mix examples from the episodic memory?